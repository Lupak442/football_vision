# 라이브러리 가져오기
import numpy as np
import pandas as pd

import cv2
import skimage
from PIL import Image
from ultralytics import YOLO
from sklearn.metrics import mean_squared_error

import json
import yaml
import time

# 전술 지도 키포인트 위치 사전 가져오기
json_path = r"pitch map labels position.json"
with open(json_path, "r", encoding="utf-8") as f:
    keypoints_map_pos = json.load(f)

# 축구장의 키포인트를 숫자에서 알파벳순으로 매핑하기
yaml_path = r"config pitch dataset.yaml"
with open(yaml_path, "r", encoding="utf-8") as file:
    classes_names_dic = yaml.safe_load(file)
classes_names_dic = classes_names_dic["names"]

# 축구장의 키포인트를 숫자에서 알파벳순으로 매핑하기
yaml_path = r"config players dataset.yaml"
with open(yaml_path, "r", encoding="utf-8") as file:
    labels_dic = yaml.safe_load(file)
labels_dic = labels_dic["names"]

print("Known coordinates of each keypoint on the tactical map:")
print(pd.DataFrame(keypoints_map_pos, index=["x", "y"]))
print("Numerical label of field keypoints (as defined when training the Yolo model):")
print(
    pd.Series(classes_names_dic, name="alpha_label")
    .reset_index()
    .rename({"index": "num_label"}, axis=1)
    .set_index("alpha_label")
    .transpose()
)
print(
    "Numerical label of the player, referee, and ball objects (as defined when training the Yolo model):"
)
print(
    pd.Series(labels_dic, name="alpha_label")
    .reset_index()
    .rename({"index": "num_label"}, axis=1)
    .set_index("alpha_label")
    .transpose()
)
print(
    "\033[1mThe dataframe representation are not used in what follows (original dictionary will be used)"
)

# 비디오 경로 설정
video_path = "project video.mp4"

# Read tactical map image
tac_map = cv2.imread("tactical map.jpg")

# 팀 색상 정의(선택한 비디오 기준)
nbr_team_colors = 2
colors_dic = {
    "Croatia": [
        (241, 246, 249),
        (235, 255, 165),
    ],  # 첼시 색상 (선수 키트 색상, GK 키트 색상)
    "France": [
        (28, 41, 21),
        (240, 241, 79),
    ],  # 맨시티 색상 (선수 키트 색상, GK 키트 색상)
}

colors_list = (
    colors_dic["Croatia"] + colors_dic["France"]
)  # 감지된 플레이어 팀 예측에 사용할 색상 목록을 정의합니다.
color_list_lab = [
    skimage.color.rgb2lab([i / 255 for i in c]) for c in colors_list
]  # color_list를 L*a*b* 공간으로 변환

# YOLOv8 플레이어 감지 모델 로드
model_players = YOLO("models/Yolo8L Players/weights/best.pt")

# YOLOv8 필드 키포인트 감지 모델 로드
model_keypoints = YOLO("models/Yolo8M Field Keypoints/weights/best copy.pt")

# 비디오 파일 열기
cap = cv2.VideoCapture(video_path)
# 캡처 객체가 올바르게 열렸는지 확인
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 128)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 72)

# 프레임 카운터 초기화
frame_nbr = 0

# 키포인트 평균 변위 허용 수준 설정(픽셀 단위) [항상 호모그래피 행렬을 업데이트하려면 -1로 설정]
keypoints_displacement_mean_tol = 10

# 플레이어 및 필드 키포인트 감지에 대한 신뢰도 임계값 설정
player_model_conf_thresh = 0.40
keypoints_model_conf_thresh = 0.70

# 마지막 프레임을 처리한 시간을 기록하도록 변수를 설정합니다.
prev_frame_time = 0
# 현재 프레임을 처리한 시간을 기록하도록 변수를 설정합니다.
new_frame_time = 0

# 공의 트랙 기록을 저장합니다.
ball_track_history = {"src": [], "dst": []}

# 공이 감지되지 않은 연속 프레임 수를 계산합니다.
nbr_frames_no_ball = 0
# 공 트랙(프레임)을 재설정하기 위한 공이 없는 프레임 수에 대한 임계값
nbr_frames_no_ball_thresh = 30
# 공 추적을 위한 거리 임계값(픽셀)
ball_track_dist_thresh = 100
# 최대 볼 트랙 길이(감지)
max_track_length = 35

# 비디오 프레임을 반복합니다.
while cap.isOpened():

    # 프레임 카운터 업데이트
    frame_nbr += 1

    # 비디오에서 프레임 읽기
    success, frame = cap.read()
    # frame = cv2.resize(frame, (1280, 720))

    # 새로운 프레임마다 전술 지도 이미지 재설정
    tac_map_copy = tac_map.copy()

    # 공 트랙 재설정
    if nbr_frames_no_ball > nbr_frames_no_ball_thresh:
        ball_track_history["dst"] = []
        ball_track_history["src"] = []

    # 성공적으로 읽힌 경우 프레임을 처리합니다.
    if success:
        # 이전 프레임이 없는 경우에 대한 예외 처리
        if frame_nbr == 1:
            detected_ball_src_pos = None

        #################### Part 1 ####################
        # Object Detection & Coordiante Transofrmation #
        ################################################

        # 프레임에서 YOLOv8 플레이어 추론 실행
        results_players = model_players(frame, conf=player_model_conf_thresh)
        # 프레임에서 YOLOv8 필드 키포인트 추론 실행
        results_keypoints = model_keypoints(frame, conf=keypoints_model_conf_thresh)

        ## 탐지 정보 추출
        bboxes_p = (
            results_players[0].boxes.xyxy.cpu().numpy()
        )  # 감지된 선수, 심판 및 공(x,y,x,y) 경계 상자
        bboxes_p_c = (
            results_players[0].boxes.xywh.cpu().numpy()
        )  # 감지된 선수, 심판 및 공(x,y,w,h) 경계 상자
        labels_p = list(
            results_players[0].boxes.cls.cpu().numpy()
        )  # 감지된 선수, 심판 및 공 라벨 목록
        confs_p = list(
            results_players[0].boxes.conf.cpu().numpy()
        )  # 감지된 선수, 심판 및 공 신뢰도 수준

        bboxes_k = (
            results_keypoints[0].boxes.xyxy.cpu().numpy()
        )  # 감지된 필드 키포인트(x,y,w,h) 경계 상자
        bboxes_k_c = (
            results_keypoints[0].boxes.xywh.cpu().numpy()
        )  # 감지된 필드 키포인트(x,y,w,h) 경계 상자
        labels_k = list(
            results_keypoints[0].boxes.cls.cpu().numpy()
        )  # 감지된 필드 키포인트 라벨 목록

        # 감지된 숫자 라벨을 알파벳 라벨로 변환
        detected_labels = [classes_names_dic[i] for i in labels_k]

        # 현재 프레임에서 감지된 필드 키포인트 좌표를 추출합니다.
        detected_labels_src_pts = np.array(
            [
                list(np.round(bboxes_k_c[i][:2]).astype(int))
                for i in range(bboxes_k_c.shape[0])
            ]
        )

        # 전술 지도에서 감지된 필드 키포인트 좌표를 가져옵니다.
        detected_labels_dst_pts = np.array(
            [keypoints_map_pos[i] for i in detected_labels]
        )

        ## 4개 이상의 키포인트가 감지되면 호모그래피 변환 행렬을 계산합니다.
        if len(detected_labels) > 3:
            # 항상 첫 번째 프레임에서 호모그래피 행렬을 계산합니다.
            if frame_nbr > 1:
                # 이전 프레임과 현재 프레임 사이에서 공통적으로 감지된 필드 키포인트를 결정합니다.
                common_labels = set(detected_labels_prev) & set(detected_labels)
                # 최소 4개의 공통 키포인트가 감지되면 특정 허용 수준을 넘어 평균적으로 변위되는지 확인합니다.
                if len(common_labels) > 3:
                    common_label_idx_prev = [
                        detected_labels_prev.index(i) for i in common_labels
                    ]  # 이전 프레임에서 공통적으로 감지된 키포인트의 레이블 인덱스 가져오기
                    common_label_idx_curr = [
                        detected_labels.index(i) for i in common_labels
                    ]  # 현재 프레임에서 공통적으로 감지된 키포인트의 레이블 인덱스 가져오기
                    coor_common_label_prev = detected_labels_src_pts_prev[
                        common_label_idx_prev
                    ]  # 이전 프레임에서 공통적으로 감지된 키포인트의 레이블 좌표를 가져옵니다.
                    coor_common_label_curr = detected_labels_src_pts[
                        common_label_idx_curr
                    ]  # 현재 프레임에서 공통적으로 감지된 키포인트의 레이블 좌표를 가져옵니다.
                    coor_error = mean_squared_error(
                        coor_common_label_prev, coor_common_label_curr
                    )  # 이전 및 현재 공통 키포인트 좌표 간의 오류를 계산합니다.
                    update_homography = (
                        coor_error > keypoints_displacement_mean_tol
                    )  # 오류가 사전 정의된 허용 수준을 초과했는지 확인
                else:
                    update_homography = True
            else:
                update_homography = True

            if update_homography:
                h, mask = cv2.findHomography(
                    detected_labels_src_pts,  # 호모그래피 행렬 계산
                    detected_labels_dst_pts,
                )

            detected_labels_prev = (
                detected_labels.copy()
            )  # 다음 프레임을 위해 현재 감지된 키포인트 라벨을 저장합니다.
            detected_labels_src_pts_prev = (
                detected_labels_src_pts.copy()
            )  # 다음 프레임을 위해 현재 감지된 키포인트 좌표를 저장합니다.

            bboxes_p_c_0 = bboxes_p_c[
                [i == 0 for i in labels_p], :
            ]  # 감지된 플레이어(라벨 0)의 경계 상자 정보(x,y,w,h)를 가져옵니다.
            bboxes_p_c_2 = bboxes_p_c[
                [i == 2 for i in labels_p], :
            ]  # 감지된 공의 경계 상자 정보(x,y,w,h) 가져오기(라벨 2)

            # 프레임에서 감지된 플레이어의 좌표를 가져옵니다(x_cencter, y_center+h/2)
            detected_ppos_src_pts = (
                bboxes_p_c_0[:, :2]
                + np.array(
                    [[0] * bboxes_p_c_0.shape[0], bboxes_p_c_0[:, 3] / 2]
                ).transpose()
            )
            # 처음 감지된 공의 좌표를 가져옵니다(x_center, y_center)
            detected_ball_src_pos = (
                bboxes_p_c_2[0, :2] if bboxes_p_c_2.shape[0] > 0 else None
            )

            # 계산된 호모그래피 행렬을 사용하여 플레이어 좌표를 프레임 평면에서 전술 지도 평면으로 변환합니다.
            pred_dst_pts = []  # 플레이어 전술 지도 좌표 목록 초기화
            for pt in detected_ppos_src_pts:  # 플레이어 프레임 좌표에 대한 루프
                pt = np.append(
                    np.array(pt), np.array([1]), axis=0
                )  # 동종 좌표계로 은닉
                dest_point = np.matmul(h, np.transpose(pt))  # 호모그래피 변환 적용
                dest_point = dest_point / dest_point[2]  # 2D 좌표로 되돌리기
                pred_dst_pts.append(
                    list(np.transpose(dest_point)[:2])
                )  # 플레이어 전술 지도 좌표 목록 업데이트
            pred_dst_pts = np.array(pred_dst_pts)

            # 계산된 호모그래피 행렬을 사용하여 공 좌표를 프레임 평면에서 전술 지도 평면으로 변환합니다.
            if detected_ball_src_pos is not None:
                pt = np.append(np.array(detected_ball_src_pos), np.array([1]), axis=0)
                dest_point = np.matmul(h, np.transpose(pt))
                dest_point = dest_point / dest_point[2]
                detected_ball_dst_pos = np.transpose(dest_point)

                # 트랙볼 위치 기록 업데이트
                if len(ball_track_history["src"]) > 0:
                    if (
                        np.linalg.norm(
                            detected_ball_src_pos - ball_track_history["src"][-1]
                        )
                        < ball_track_dist_thresh
                    ):
                        ball_track_history["src"].append(
                            (
                                int(detected_ball_src_pos[0]),
                                int(detected_ball_src_pos[1]),
                            )
                        )
                        ball_track_history["dst"].append(
                            (
                                int(detected_ball_dst_pos[0]),
                                int(detected_ball_dst_pos[1]),
                            )
                        )
                    else:
                        ball_track_history["src"] = [
                            (
                                int(detected_ball_src_pos[0]),
                                int(detected_ball_src_pos[1]),
                            )
                        ]
                        ball_track_history["dst"] = [
                            (
                                int(detected_ball_dst_pos[0]),
                                int(detected_ball_dst_pos[1]),
                            )
                        ]
                else:
                    ball_track_history["src"].append(
                        (int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1]))
                    )
                    ball_track_history["dst"].append(
                        (int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1]))
                    )
            # 트랙이 임계값을 초과하는 경우 가장 오래된 추적된 공 위치 제거
            if len(ball_track_history) > max_track_length:
                ball_track_history["src"].pop(0)
                ball_track_history["dst"].pop(0)

        ######### Part 2 ##########
        # Players Team Prediction #
        ###########################

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 프레임을 RGB로 변환
        obj_palette_list = []  # 플레이어 색상 팔레트 목록 초기화
        palette_interval = (
            0,
            5,
        )  # 주요 색상 팔레트에서 추출할 색상 간격(1~5번째 색상)
        annotated_frame = frame  # 주석이 달린 프레임 만들기

        ## 감지된 플레이어(레이블 0)를 반복하고 정의된 간격에 따라 주요 색상 팔레트를 추출합니다.
        for i, j in enumerate(list(results_players[0].boxes.cls.cpu().numpy())):
            if int(j) == 0:
                bbox = (
                    results_players[0].boxes.xyxy.cpu().numpy()[i, :]
                )  # bbox 정보 가져오기(x,y,x,y)
                obj_img = frame_rgb[
                    int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])
                ]  # 프레임에서 bbox 자르기
                obj_img_w, obj_img_h = obj_img.shape[1], obj_img.shape[0]
                center_filter_x1 = np.max([(obj_img_w // 2) - (obj_img_w // 5), 1])
                center_filter_x2 = (obj_img_w // 2) + (obj_img_w // 5)
                center_filter_y1 = np.max([(obj_img_h // 3) - (obj_img_h // 5), 1])
                center_filter_y2 = (obj_img_h // 3) + (obj_img_h // 5)
                center_filter = obj_img[
                    center_filter_y1:center_filter_y2, center_filter_x1:center_filter_x2
                ]
                obj_pil_img = Image.fromarray(
                    np.uint8(center_filter)
                )  # pillow image 변환

                reduced = obj_pil_img.convert(
                    "P", palette=Image.Palette.WEB
                )  # 웹 팔레트로 변환(216색)
                palette = (
                    reduced.getpalette()
                )  # 팔레트를 [r,g,b,r,g,b,...]로 가져옵니다.
                palette = [
                    palette[3 * n : 3 * n + 3] for n in range(256)
                ]  # 3을 3으로 그룹화 = [[r,g,b],[r,g,b],...]
                color_count = [
                    (n, palette[m]) for n, m in reduced.getcolors()
                ]  # 빈도에 따라 팔레트 색상 목록 만들기
                RGB_df = (
                    pd.DataFrame(color_count, columns=["cnt", "RGB"])
                    .sort_values(  # 정의된 팔레트 간격을 기반으로 데이터프레임 생성
                        by="cnt", ascending=False
                    )
                    .iloc[palette_interval[0] : palette_interval[1], :]
                )
                palette = list(RGB_df.RGB)  # 팔레트를 목록으로 변환(빠른 처리를 위해)
                annotated_frame = cv2.rectangle(
                    annotated_frame,  # 중앙 필터 bbox 주석 추가
                    (int(bbox[0]) + center_filter_x1, int(bbox[1]) + center_filter_y1),
                    (int(bbox[0]) + center_filter_x2, int(bbox[1]) + center_filter_y2),
                    (0, 0, 0),
                    2,
                )

                # 감지된 플레이어 색상 팔레트 목록 업데이트
                obj_palette_list.append(palette)

        ## 감지된 모든 플레이어 색상 팔레트의 각 색상과 사전 정의된 팀 색상 사이의 거리를 계산합니다.
        players_distance_features = []
        # 감지된 플레이어 추출된 색상 팔레트를 반복합니다.
        for palette in obj_palette_list:
            palette_distance = []
            palette_lab = [
                skimage.color.rgb2lab([i / 255 for i in color]) for color in palette
            ]  # 색상을 L*a*b* 공간으로 변환
            # 팔레트의 색상을 반복합니다.
            for color in palette_lab:
                distance_list = []
                # 미리 정의된 팀 색상 목록을 반복합니다.
                for c in color_list_lab:
                    # distance = np.linalg.norm([i/255 - j/255 for i,j in zip(color,c)])
                    distance = skimage.color.deltaE_cie76(
                        color, c
                    )  # Lab 색상 공간에서 유클리드 거리 계산
                    distance_list.append(distance)  # 현재 색상의 거리 목록 업데이트
                palette_distance.append(
                    distance_list
                )  # 현재 팔레트의 거리 목록 업데이트
            players_distance_features.append(
                palette_distance
            )  # 거리 특징 목록 업데이트

        ## 거리 특성을 기반으로 감지된 선수 팀 예측
        players_teams_list = []
        # 플레이어 거리 기능에 대한 루프
        for distance_feats in players_distance_features:
            vote_list = []
            # 각 색상의 거리에 대한 루프
            for dist_list in distance_feats:
                team_idx = (
                    dist_list.index(min(dist_list)) // nbr_team_colors
                )  # 최소 거리를 기준으로 현재 색상에 대한 팀 인덱스 할당
                vote_list.append(team_idx)  # 현재 색상 팀 예측으로 vote_list 업데이트
            players_teams_list.append(
                max(vote_list, key=vote_list.count)
            )  # 투표 집계를 통해 현재 플레이어 팀을 예측합니다.

        #################### Part 3 #####################
        # 주석이 포함된 업데이트된 프레임 및 전술 지도 #
        #################################################

        ball_color_bgr = (0, 0, 255)  # 전술 지도의 공 주석 색상(GBR)
        j = 0  # 감지된 플레이어의 카운터 초기화 중
        palette_box_size = 10  # 색상 상자 크기를 픽셀 단위로 설정(표시용)

        # 플레이어 감지 모델을 통해 감지된 모든 객체를 반복합니다.
        for i in range(bboxes_p.shape[0]):
            conf = confs_p[i]  # 현재 감지된 객체에 대한 신뢰도 확보
            if labels_p[i] == 0:  # 감지된 플레이어에 대한 주석 표시 (label 0)

                # 탐지된 각 플레이어에 대해 추출된 색상 팔레트 표시
                # palette = obj_palette_list[j]  # 탐지된 플레이어의 색상 팔레트 가져오기
                # for k, c in enumerate(palette):
                #     c_bgr = c[::-1]  # 색상을 BGR로 변환
                #     annotated_frame = cv2.rectangle(
                #         annotated_frame,
                #         (
                #             int(bboxes_p[i, 2]) + 3,  # 프레임에 색상 팔레트 주석 추가
                #             int(bboxes_p[i, 1]) + k * palette_box_size,
                #         ),
                #         (
                #             int(bboxes_p[i, 2]) + palette_box_size,
                #             int(bboxes_p[i, 1]) + (palette_box_size) * (k + 1),
                #         ),
                #         c_bgr,
                #         -1,
                #     )

                team_name = list(colors_dic.keys())[
                    players_teams_list[j]
                ]  # 탐지된 선수 팀 예측 가져오기
                color_rgb = colors_dic[team_name][0]  # 탐지된 플레이어 팀 색상 가져오기
                color_bgr = color_rgb[::-1]  # 색상을 bgr로 변환

                annotated_frame = cv2.rectangle(
                    annotated_frame,
                    (
                        int(bboxes_p[i, 0]),
                        int(bboxes_p[i, 1]),
                    ),  # 팀 색상으로 bbox 주석 추가
                    (int(bboxes_p[i, 2]), int(bboxes_p[i, 3])),
                    color_bgr,
                    1,
                )

                cv2.putText(
                    annotated_frame,
                    team_name + f" {conf:.2f}",  # 팀 이름 주석 추가
                    (int(bboxes_p[i, 0]), int(bboxes_p[i, 1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color_bgr,
                    2,
                )

                # 필드 키포인트가 3개 이상 감지된 경우 전술 맵 플레이어 게시물 색상 코드화 주석 추가
                if len(detected_labels_src_pts) > 3:
                    tac_map_copy = cv2.circle(
                        tac_map_copy,
                        (int(pred_dst_pts[j][0]), int(pred_dst_pts[j][1])),
                        radius=5,
                        color=color_bgr,
                        thickness=-1,
                    )

                j += 1  # 플레이어 카운터 업데이트
            else:  # 다른 탐지에 대한 주석 표시(라벨 1, 2)
                annotated_frame = cv2.rectangle(
                    annotated_frame,
                    (
                        int(bboxes_p[i, 0]),
                        int(bboxes_p[i, 1]),
                    ),  # 흰색 상자 주석 추가
                    (int(bboxes_p[i, 2]), int(bboxes_p[i, 3])),
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    annotated_frame,
                    labels_dic[labels_p[i]]
                    + f" {conf:.2f}",  # 흰색 레이블 텍스트 주석 추가
                    (int(bboxes_p[i, 0]), int(bboxes_p[i, 1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

                # 탐지된 경우 전술 맵 볼 게시 주석 추가
                if detected_ball_src_pos is not None:
                    tac_map_copy = cv2.circle(
                        tac_map_copy,
                        (int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1])),
                        radius=5,
                        color=ball_color_bgr,
                        thickness=3,
                    )
        for i in range(bboxes_k.shape[0]):
            annotated_frame = cv2.rectangle(
                annotated_frame,
                (
                    int(bboxes_k[i, 0]),
                    int(bboxes_k[i, 1]),
                ),  # 팀 색상으로 bbox 주석 추가
                (int(bboxes_k[i, 2]), int(bboxes_k[i, 3])),
                (0, 0, 0),
                1,
            )

        # 전술 지도에 볼 트랙을 표시합니다
        if len(ball_track_history["src"]) > 0:
            points = (
                np.hstack(ball_track_history["dst"])
                .astype(np.int32)
                .reshape((-1, 1, 2))
            )
            tac_map_copy = cv2.polylines(
                tac_map_copy, [points], isClosed=False, color=(0, 0, 100), thickness=2
            )

        # 주석이 달린 프레임과 전술 지도를 하나의 이미지에 색 테두리 분리로 결합
        border_color = [255, 255, 255]  # 테두리 색상 설정(BGR)
        annotated_frame = cv2.copyMakeBorder(
            annotated_frame,
            40,
            10,
            10,
            10,  # 주석이 달린 프레임에 테두리 추가
            cv2.BORDER_CONSTANT,
            value=border_color,
        )
        tac_map_copy = cv2.copyMakeBorder(
            tac_map_copy,
            70,
            50,
            10,
            10,
            cv2.BORDER_CONSTANT,  # 전술 맵에 경계 추가
            value=border_color,
        )
        tac_map_copy = cv2.resize(
            tac_map_copy, (tac_map_copy.shape[1], annotated_frame.shape[0])
        )  # 전술 맵 크기 조정
        final_img = cv2.hconcat((annotated_frame, tac_map_copy))  # 두 이미지 모두 연결
        ## 정보 주석 추가
        cv2.putText(
            final_img,
            "Tactical Map",
            (1370, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            final_img,
            "Press 'p' to pause & 'q' to quit",
            (820, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            2,
        )

        new_frame_time = time.time()  # 현재 프레임 처리 완료 후 시간 확보
        fps = 1 / (
            new_frame_time - prev_frame_time
        )  # FPS를 1/(frame processing duration)로 계산합니다
        prev_frame_time = new_frame_time  # 다음 프레임에서 사용할 현재 시간 저장
        cv2.putText(
            final_img,
            "FPS: " + str(int(fps)),
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            2,
        )

        # 최종 주석이 달린 프레임 표시
        cv2.imshow(
            "YOLOv8 Players and Field Keypoints Detection with Team Prediction and Tactical Map",
            final_img,
        )

        # 키보드 사용자 입력 처리("일시정지/일시정지 해제 시 p", 종료 시 "q")
        key = cv2.waitKey(1)
        # 'q'를 누르면 루프가 끊어집니다
        if key == ord("q"):
            break
        if key == ord("p"):
            cv2.waitKey(-1)  # 키를 누를 때까지 기다리다
    else:
        # 비디오의 끝에 도달하면 루프를 끊습니다
        break

# 비디오 캡처 개체를 해제하고 디스플레이 창을 닫습니다cap.release()
cv2.destroyAllWindows()

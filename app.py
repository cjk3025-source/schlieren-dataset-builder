import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import zipfile
from io import BytesIO

# --- 1. 기본 설정 ---
st.set_page_config(page_title="Schlieren Dataset Builder", layout="wide")
st.title("📦 Schlieren Dataset Builder v2.0")
st.markdown("전처리된 슐리렌 영상(Fume/Spatter)과 인장시험 기반 양/불 정답지를 팀원에게 전달하기 위한 패키징 툴입니다.")

# --- 2. 사이드바: 메타데이터 및 정답지 입력 ---
st.sidebar.header("📝 1. 시편 정보 및 정답지 입력")
specimen_id = st.sidebar.text_input("시편 번호 (예: SP_045)", "SP_045")
weld_type = st.sidebar.selectbox("용접 종류", ["맞대기(Butt) 용접", "겹치기(Lap) 용접"])

st.sidebar.subheader("🎯 최종 품질 판정")
quality_label = st.sidebar.radio(
    "인장시험 등 물리적 평가 결과", 
    ["🟢 정상 (Pass)", "🔴 불량 (Fail)"]
)

# --- 3. 메인 화면: 영상 업로드 및 전처리 ---
st.header("⚙️ 2. 슐리렌 영상 업로드 및 전처리 확인")
uploaded_video = st.file_uploader("원본 슐리렌 영상 업로드 (mp4, avi)", type=['mp4', 'avi'])

# 파라미터 조절 슬라이더
col1, col2 = st.columns(2)
with col1:
    fume_threshold = st.slider("Fume 추출 임계값", 0, 255, 30)
with col2:
    spatter_threshold = st.slider("Spatter 추출 임계값", 0, 255, 200)

if uploaded_video is not None:
    st.success(f"{uploaded_video.name} 파일 업로드 완료! (UI 시각화 및 패키징 대기 중)")
    
    # 💡 실제 구현 시: 여기에 cv2.VideoCapture를 연결해 첫 프레임이나 짧은 구간의 
    # 원본/Fume 마스크/Spatter 마스크를 st.image()로 띄워주는 코드가 들어갑니다.
    st.info("여기에 영상의 첫 프레임을 분석한 Fume/Spatter 마스킹 프리뷰가 표시됩니다.")

# --- 4. 데이터 패키징 및 추출 ---
st.header("📤 3. ML 팀원 전달용 데이터셋 패키징")
st.markdown("입력된 정답지와 전처리 데이터를 하나의 ZIP 파일로 묶어 추출합니다.")

if st.button("📦 데이터셋 패키징 및 다운로드 생성"):
    if uploaded_video is None:
        st.warning("먼저 영상을 업로드해주세요!")
    else:
        with st.spinner("데이터를 정제하고 압축하는 중입니다..."):
            # 1. 메타데이터 CSV 생성
            label_value = "Pass" if "정상" in quality_label else "Fail"
            df_meta = pd.DataFrame({
                "specimen_id": [specimen_id],
                "weld_type": [weld_type],
                "fume_threshold_used": [fume_threshold],
                "spatter_threshold_used": [spatter_threshold],
                "quality_label": [label_value],
                "remarks": [f"{specimen_id} {weld_type}은(는) {label_value}입니다."]
            })
            
            # 2. 메모리 상에서 ZIP 파일 생성 (가상 파일)
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                # CSV 파일 추가
                zip_file.writestr(f"{specimen_id}_metadata.csv", df_meta.to_csv(index=False))
                # 업로드된 원본(또는 전처리 완료된) 영상 추가
                zip_file.writestr(f"{specimen_id}_preprocessed.mp4", uploaded_video.getvalue())
                # 💡 실제 구현 시: 추출된 마스크 이미지 폴더도 여기에 반복문으로 추가
            
            # 3. 다운로드 버튼 생성
            st.success("✅ 패키징 완료! 아래 버튼을 눌러 팀원에게 전달할 파일을 다운로드하세요.")
            st.download_button(
                label="📥 Dataset ZIP 다운로드",
                data=zip_buffer.getvalue(),
                file_name=f"dataset_{specimen_id}.zip",
                mime="application/zip"
            )
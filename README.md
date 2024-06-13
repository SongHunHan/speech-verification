# Speech Verification Repository

이 저장소는 음성 데이터를 기반으로 화자 인식 모델을 학습하고 사용하는 방법을 제공합니다. 한국어 음성 데이터셋인 AIHub의 화자 인식용 음성 데이터셋을 사용하여 학습을 진행했습니다.

## 데이터셋

- AIHub 화자 인식용 음성 데이터
- 데이터셋 링크: [AIHub 화자 인식 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=537)


## 모델 학습

1. 데이터셋을 준비합니다. AIHub에서 화자 인식 데이터셋을 다운로드하고, 데이터셋 경로를 `config` 폴더의 해당 모델 설정 파일(예: `wav2vec.yaml`)에 지정합니다.

2. 학습에 필요한 하이퍼파라미터를 `config` 폴더의 해당 모델 설정 파일에서 조정합니다. 학습 에폭 수, 배치 크기, 학습률 등을 설정할 수 있습니다.

3. 다음 명령어를 실행하여 모델을 학습합니다:

python train.py --config config/wav2vec.yaml


학습된 모델은 finetuned_model 폴더에 생성됩니다.


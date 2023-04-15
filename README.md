# Gas Energy Document Summarization

## Competition Information
![image](https://user-images.githubusercontent.com/72617445/232245131-290c8c85-529c-4562-ad97-ba6a00009946.png)



- submisson.py : 요약문 생성하여 submisson 파일 생성
- dataset.py : ai hub 문서요약 데이터셋 .tsv로 변환
- gas_to_submission.py : 가스문건만 법률 파인튜닝 모델로 생성하기 위함
- rouge_score.py : rouge_score 측정

## 모델설명
- kobart_summary_epoch15 : huggingface에 있는 gogamza/kobart-summarization 모델을 base로 뉴스요약문으로 16epoch fine-tuning
- kobart_law : kobart_summary_epoch15 모델로 법률요약문으로 7epoch fine-tunning

## How to fine-tuning
## Install KoBART
- pip install git+https://github.com/SKT-AI/KoBART#egg=kobart

## Requirements
- pytorch==1.7.1
- transformers==4.3.3
- pytorch-lightning==1.1.0
- pyyaml==5.4.1

## Data
- AI hub 문서요약 데이터셋 중 뉴스기사, 법률만 활용
- dataset.py 를 통해 KoBART-summarization/data 안에 tsv 파일로 저장
  
|  news  | summary |
|  원문   |  요약문   |  

## Fine-tuning
- huggingface에 있는 gogamza/kobart-summarization 모델을 base로 fine-tuning함

- pip install -r requirements.txt

[use gpu(1080TI X 4)]
- python train.py --gradient_clip_val 1.0 --max_epochs 50 --default_root_dir logs --gpus 4 --batch_size 6 --num_workers 8 --accelerator ddp --max_len 512

## Generation Model
- pytorch-lightning binary --> huggingface binary로 추출 작업 필요
- hparams의 경우에는 KoBART-summarization/logs/tb_logs/default/version_0/hparams.yaml 파일을 활용
- model_binary 의 경우에는 KoBART-summarization/logs/kobart_summary-model_chp 안에 있는 .ckpt 파일을 활용
- 변환 코드를 실행하면 KoBART-summarization/kobart_summary 에 model binary 가 추출 됨

- python get_model_binary.py --hparams hparams.yaml --model_binary epoch=15-val_loss=6.178.ckpt

## Making Submission File
- data 폴더 안에 저장됨
- python submission.py

## Reference
- [KoBART](https://github.com/SKT-AI/KoBART)
- https://github.com/nlee0212/KoBART-summarization

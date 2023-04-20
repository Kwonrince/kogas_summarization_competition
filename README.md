# Gas Energy Document Summarization

한국가스공사에서 주최한 제 3회 빅데이터·인공지능 경진대회에서 예비창업가 부문 최우수상을 수상한 '비정형팀플' 팀의 자료입니다. SKT-AI에서 배포한 [KoBART](https://github.com/SKT-AI/KoBART)를 AI Hub에서 제공하는 [문서요약 텍스트](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=97)로 Fine Tuning을 진행했습니다. (신문기사 약 30만건, 법률문서 약 3만건 사용)

발표자료 및 자세한 내용은 [발표자료](./slides/slides.pdf)에서 확인할 수 있습니다.


## File Description
- `./KoBART-summarization/` : 학습 코드 폴더
- `dataset.py` : ai hub 문서요약 데이터셋 .tsv로 변환
- `submisson.py` : 요약문 생성하여 submisson 파일 생성
- `gas_to_submission.py` : 가스문건에 해당하는 부분만 법률용 모델로 생성한 요약문으로 대체
- `rouge_score.py` : rouge_score 측정

## Model Description
- `./KoBART-summarization/kobart_summary_epoch15` : huggingface에 있는 gogamza/kobart-summarization 모델을 신문기사 데이터셋으로 16epoch fine-tuning
- `./KoBART-summarization/kobart_law` : kobart_summary_epoch15 모델로 법률요약문으로 7epoch fine-tunning
- 학습된 모델 파일은 용량문제로 업로드 하지 않음

# How to fine-tuning
## Requirements
- pytorch==1.7.1
- transformers==4.3.3
- pytorch-lightning==1.1.0
- pyyaml==5.4.1

```bash
$ cd ./KoBART-summarization
$ pip install -r requirements.txt
```

## Install KoBART
```bash
$ pip install git+https://github.com/SKT-AI/KoBART#egg=kobart
```

## Prepare Dataset
- AI hub 문서요약 데이터셋 중 뉴스기사, 법률만 활용 [다운로드](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=97)
- `./dataset.py` 를 통해 `./KoBART-summarization/data/` 안에 tsv 파일로 저장
- AI hub 데이터셋 저장 경로
```bash
── ./train_data
  ├──  Training
  │    ├──  법률_train_original.json
  │    ├──  사설잡지_train_original.json
  │    ├──  신문기사_train_original.json
  └──  Validation
       ├──  법률_valid_original.json
       ├──  사설잡지_valid_original.json
       └──  신문기사_valid_original.json
```

```bash
$ python dataset.py
```
  
|  news  | summary |<br/>
|  원문   |  요약문  |  

## Fine-tuning
- huggingface에 있는 gogamza/kobart-summarization 모델을 base로 fine-tuning함
- GPU (1080TI X 4)

```bash
$ python train.py --gradient_clip_val 1.0 --max_epochs 50 --default_root_dir logs --gpus 4 --batch_size 6 --num_workers 8 --accelerator ddp --max_len 512
```

## Generation
- pytorch-lightning binary --> huggingface binary로 추출 작업 필요
- hparams의 경우에는 KoBART-summarization/logs/tb_logs/default/version_0/hparams.yaml 파일을 활용
- model_binary 의 경우에는 KoBART-summarization/logs/kobart_summary-model_chp 안에 있는 .ckpt 파일을 활용
- 변환 코드를 실행하면 KoBART-summarization/kobart_summary 에 model binary 가 추출 됨

```bash
# example
$ python get_model_binary.py --hparams hparams.yaml --model_binary epoch=15-val_loss=6.178.ckpt
```

## Making Submission File
- `./submission` 폴더 안에 저장됨

```bash
$ python submission.py
```

## Reference
- [KoBART](https://github.com/SKT-AI/KoBART)
- https://github.com/nlee0212/KoBART-summarization

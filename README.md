# Paper_abstract_summarization

AI hub 논문 요약 데이터를 이용하여 생성 요약을 수행하는 레포입니다.

# To-Do

- [x] 학습 code pipeline 구축 - Huggingface Trainer 사용 / Trainer 미사용 version
- [x] Mecab을 이용한 usable token 비교 code 구현
- [x] [RL를 이용한 policy learning](https://arxiv.org/pdf/1705.04304.pdf) code 구현
- [ ] Rerank 구현
- [ ] multilingual(한국어, 영어 위주) 모델 fine-tuning or pretrain 완료

---

# Doing

* Rerank 구현

---

* RL를 이용한 policy learning

: 좀 더 Rouge 및 정성적인 면의 평가방법에 대해 직접적으로 loss에 적용하기 위해서

* Rerank

: beam search 시, Top 1이 정답에 가까운 경우가 약 25%로 나머지 순위에도 많이 정답에 가까운 경우가 존재

# Workflows — 한눈에 보는 흐름도

> 프로젝트의 모든 흐름을 **다이어그램**으로 정리했습니다. GitHub에서 이 파일을 열면 아래 Mermaid 블록이 **그림으로 자동 렌더링**됩니다.
> 개념 설명은 [`defense-qa-ko.md`](defense-qa-ko.md) / [`defense-qa-en.md`](defense-qa-en.md), 수학 심화는 [`defense-guide.md`](defense-guide.md) 참고.

---

## 1. 전체 파이프라인 (raw 신호 → 점수)

원본 EEG가 점수가 되기까지의 큰 흐름입니다.

```mermaid
flowchart LR
  A["PhysioNet EDF<br/>64ch x 160Hz"] --> B["load_raw<br/>(data.py)"]
  B --> C["filter_raw<br/>avg-ref + 7-30Hz FIR<br/>(preprocessing.py)"]
  C --> D["epochs_from_raw<br/>2s windows at T1/T2<br/>~45 x (64 x 321)"]
  D --> E["MyCSP<br/>64x321 to 4 numbers<br/>(csp.py)"]
  E --> F["LDA<br/>4 numbers to class"]
  F --> G["cross_val_score<br/>honest accuracy"]
  G --> H["60% gate<br/>109 x 6 mean = 0.658"]
```

---

## 2. 모듈 구조 (어떤 파일이 무엇을 부르나)

CLI 입구부터 각 모듈이 호출되는 관계. 점선은 보너스 경로.

```mermaid
flowchart TD
  CLI["mybci.py<br/>(CLI entry)"] --> TR["train.py"]
  CLI --> PR["predict.py"]
  CLI --> EV["evaluate.py<br/>(run_all / tune)"]

  TR --> PRE["preprocessing.build_dataset"]
  EV --> PRE
  PRE --> DATA["data.load_raw<br/>(EDF read + concat + montage)"]
  PRE --> FILT["preprocessing.filter_raw<br/>(avg-ref + 7-30Hz)"]

  TR --> PIPE["pipeline.build_pipeline"]
  EV --> PIPE
  PIPE --> CSP["csp.MyCSP<br/>(from-scratch)"]
  PIPE --> CLF["LDA"]

  PR --> JOB["models/*.joblib<br/>(saved by train)"]

  CSP -.->|bonus A| JAC["jacobi.generalized_eigh"]
  PIPE -.->|bonus F| FB["fbcsp.FilterBankCSP"]
  PIPE -.->|bonus D| OWN["own_lda.OwnLDA"]
  EXT["external.load_external<br/>(bonus G: BCI IV-2a)"] -.->|feeds same pipeline| PIPE
```

---

## 3. Train 모드 — 독립적인 두 갈래 (A 채점 / B 저장)

`train`은 **채점(A)**과 **모델 저장(B)**을 둘 다 합니다. 둘은 별개의 80/20 분할입니다.

```mermaid
flowchart TD
  X["build_dataset<br/>(X, y) ~45 trials"] --> A["A. cross_val_score<br/>ShuffleSplit x10<br/>refit CSP+LDA each fold"]
  X --> B["B. train_test_split<br/>fixed 80/20 once"]

  A --> AS["print 10 fold scores<br/>+ mean = 0.7333"]
  A -.->|10 models discarded| NUL["(thrown away)"]

  B --> BF["fit pipeline on 80%"]
  BF --> SAVE["save .joblib<br/>pipeline + held-out 20% (9 trials)"]
```

- **A** = 정직한 점수(60% 게이트가 쓰는 값), 만든 모델은 전부 버림.
- **B** = predict가 굴릴 실제 모델 + predict용 시험지 9개.

---

## 4. Predict / Realtime 모드 — 스트림 시뮬레이션

저장된 모델로, 처음 보는 9문제를 **하나씩** 예측 (이벤트당 < 2초).

```mermaid
flowchart LR
  J["models/*.joblib<br/>(model + 9 held-out)"] --> L["load once"]
  L --> S["stream 1 epoch at a time"]
  S --> P["pipe.predict<br/>CSP transform + LDA dot"]
  P --> CHK["assert latency < 2.0s<br/>(~0.5 ms each)"]
  CHK --> O["print [pred] [truth] match"]
  O --> ACC["accuracy = 6/9"]
```

---

## 5. Score — 60% 게이트 (run_all)

109명 × 6실험을 채점해 **실험 유형별 평균(6개) → 그 6개의 평균**.

```mermaid
flowchart TD
  RA["run_all<br/>109 subjects x 6 experiments"] --> LOOP["for each experiment 0..5"]
  LOOP --> SUBJ["cross_val_accuracy<br/>per subject (x109)"]
  SUBJ --> MEAN["mean over 109 subjects"]
  MEAN --> SIX["6 experiment means"]
  SIX --> GRAND["mean of the 6 = 0.658"]
  GRAND --> GATE{">= 0.60 ?"}
  GATE -->|yes| PASS["PASS (5/5)"]
  GATE -->|no| FAIL["FAIL"]
```

---

## 6. CSP 내부 (fit + transform)

직접 구현한 핵심. 공분산 → 고유분해 → 상위 4 필터 → 로그-분산.

```mermaid
flowchart TD
  XY["X (trials x 64 x 321), y"] --> COV["_class_cov<br/>C1, C2 (64x64)<br/>trace-norm + shrinkage"]
  COV --> EIG["eigh(C1, C1+C2)<br/>eigenvectors + lambda"]
  EIG --> SORT["sort by abs(lambda - 0.5)<br/>keep top 4"]
  SORT --> FILT["filters_ (4 x 64)"]
  FILT --> Z["transform: Z = filters_ @ E<br/>(4 x 321)"]
  Z --> VAR["variance per row<br/>normalize"]
  VAR --> LOG["log -> 4 numbers"]
```

- **lambda = a/(a+b)** = 그 필터 출렁임이 한 클래스로 쏠린 비율 (0.5에서 멀수록 좋음).
- 보너스: `eigh`를 자작 `jacobi.generalized_eigh`로 교체 가능 (scipy와 ~1e-14 일치).

---

## 7. 교차검증 + 누수 방지 (한 fold의 안)

매 fold가 **새로** CSP+LDA를 학습분으로만 학습 → 시험 데이터를 미리 못 봄.

```mermaid
flowchart LR
  D["45 trials"] --> R["round k of 10"]
  R --> SP["hide 9 (test) / keep 36 (train)"]
  SP --> FIT["refit CSP + LDA<br/>on the 36 ONLY"]
  FIT --> SC["score the hidden 9"]
  SC --> AVG["average all 10 -> 0.7333"]
```

- CSP가 Pipeline 첫 단계라 fold마다 재학습됨 → **누수 없음** (누수 시 1.0 vs 정상 0.844).

---

## 8. 개념 9단계 (배움의 지도)

평가·이해를 위한 개념 순서.

```mermaid
flowchart LR
  S1["1. EEG table<br/>64ch x 160Hz"] --> S2["2. goal<br/>6 binary tasks"]
  S2 --> S3["3. channels<br/>C3 / C4"]
  S3 --> S4["4. epoching<br/>~45 trials"]
  S4 --> S5["5. ERD<br/>contralateral drop"]
  S5 --> S6["6. filter<br/>7-30Hz"]
  S6 --> S7["7. CSP<br/>64x321 to 4"]
  S7 --> S8["8. LDA<br/>4 to decision"]
  S8 --> S9["9. cross-val<br/>+ 60% gate"]
```

---

## 9. 보너스 5개 → 평가표 항목 매핑

```mermaid
flowchart LR
  A["A. Jacobi eigensolver<br/>(~5e-14 vs scipy)"] --> IMPL["Implementations"]
  C["C. GridSearch tuning<br/>(leakage-free)"] --> IMPL
  D["D. OwnLDA<br/>(100% match sklearn)"] --> IMPL
  F["F. FBCSP<br/>(4 sub-bands)"] --> FE["Feature engineering"]
  F --> IMPL
  G["G. BCI IV-2a dataset<br/>(0.855)"] --> DS["Datasets"]
```

---

> **요약**: 위 1~9번 그림이 이 프로젝트의 전체 동작·구조·평가 흐름 전부입니다. 디펜스 때 화면에 띄워 두면 흐름 질문에 바로 답할 수 있어요.

# Total Perspective Vortex — 디펜스 Q&A (한국어판)

> 평가표 순서로 가는 디펜스 안내서입니다. 영어판 [`defense-qa-en.md`](defense-qa-en.md)와 구조가 1:1로 같아요.
> **개념을 따로 빼지 않고, 그 개념이 필요한 평가 항목 안에 바로** 넣었습니다 — ERD 설명을 보려고 다른 장으로 점프할 필요가 없어요. 항목마다: **📋 평가표 원문 → 📚 알아야 할 개념 → 🗣️ 할 말 / 🖥️ 시연 → ❓ 깊은 Q&A**.
> 실제 평가는 영어로 진행되므로 **📋 기준은 평가표 영어 원문 그대로** 두고 한국어 번역을 붙였습니다. 숫자는 이 코드베이스 실측. 수학 심화: [`defense-guide.md`](defense-guide.md), 그림: [`workflows.md`](workflows.md).

표기: 📋 기준 · 📚 개념 · 🗣️ 할 말 · 🖥️ 실행/보여주기 · 👉 화면 가리키기 · ❓ 질문+답.

---

# 0. 디펜스 전 준비 (한 번만)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 데이터셋을 프로젝트 폴더로 받고 60% 게이트 확인 (최초엔 느림).
python scripts/validate_60.py            # 전체 109명 → 평균 0.658
python scripts/bonus_demo.py             # 보너스 데이터셋(BCI IV-2a)도 ./mne_data로 받음
```
두 데이터셋 모두 프로젝트 안 `./mne_data/`(PhysioNet ~3.1GB, BCI IV-2a ~83MB), gitignore. 평가 컴퓨터에선 위 두 명령을 미리 한 번 (최초 다운로드는 인터넷 필요).

---

# 1. 60초 피치 + 지도

**피치:** "EEG 뇌-컴퓨터 인터페이스입니다. PhysioNet 운동/운동상상 녹음(109명, 64채널)에서 사람이 한 동작(왼손/오른손, 양손/양발, 실제/상상)을 **뇌 신호만으로** 분류합니다. 3단계: **전처리**(평균 기준, 7–30Hz 필터, 2초 에폭), **차원 축소**(64채널을 판별력 있는 4숫자로 바꾸는 **직접 구현 CSP**), **분류**(LDA). `cross_val_score`로 정직하게 채점. 필수 기준은 직접 구현한 차원 축소기를 sklearn에 통합 + 6실험 109명 평균 **≥ 60%**인데 **0.658** 달성, 보너스 5개 전부."

**지도:** `EDF 로드 → 7–30Hz 필터 → 2초 에폭 → CSP(64×321→4) → LDA → 교차검증 → 60% 게이트`.

---

# 2. 워크스루 — 평가표 항목별 (개념 포함)

## 2.0 세팅 & "이 프로젝트가 뭔가"
**🗣️ 할 말:** "제 제출 디렉터리입니다 — Python 프로젝트: 진입점 `mybci.py`, 핵심 패키지 `tpv/`, 스크립트 `scripts/`. 세팅은 venv + `pip install -r requirements.txt`; 데이터셋(~3.1GB)은 최초 실행 시 프로젝트 `mne_data/`로 자동 다운로드."
**🖥️ 시연:** `ls` → `source .venv/bin/activate` → (선택) `pytest -m "not network"`(전부 통과). 그다음 위 60초 피치.

---

## 2.1 Preprocessing — *"Watch it for the plot"*
**📋 평가 기준 (평가표 원문 그대로):**
> *"Check if the data were parsed then visualized with a script, showing raw and filtered data. The plots should look like what is shown in the video, the filtered signal being "cleaner"."*
> (번역) 데이터를 파싱한 뒤 스크립트로 raw·filtered를 시각화하는지, 필터된 게 더 "깨끗"한지 확인.

**📚 알아야 할 개념:**
- **EEG = 머리에 붙인 마이크 64개.** 64개 전극이 **초당 160번** 전압을 µV로 잼 → ~125초 녹음당 `(64, 20000)` 표. 이 그림은 그 신호의 **필터 전 vs 후**를 보여줌.
- **필터(이 그림의 핵심).** ① **평균 기준**: 매 순간 **64채널 평균을 각 채널에서 빼** 공통 성분 제거. ② **7–30Hz 밴드패스(FIR, firwin, zero-phase)**: 운동상상 대역만 남기고 느린 드리프트(<7)·근육/전기잡음(>30) 버림. *왜 7–30Hz가 의미 있는 대역인지 → Feature extraction.* 긴 연속 신호를 먼저 필터, 그다음 에폭.
- **그다음 에폭(정형 — 이 그림엔 안 나옴).** 표시 **T0(휴식), T1/T2(두 동작)** 중 휴식을 버리고 T1/T2마다 **2초 창**을 오려냄 → 한 문제 `(64 × 321)`(321 = 2초 × 160 + 1, 사람당 ~45문제). raw-vs-filtered 그림은 **연속** 신호이지 에폭이 아님.

**🗣️ 할 말:** "전처리부터 — 원본 녹음을 분석에 쓸 수 있게 바꾸는 단계입니다. 스크립트 하나로 EEG를 불러와 **두 관점에서, 각각 필터 전/후로** 보여줍니다: 먼저 **시계열**(원본 출렁임 vs 7–30Hz 필터)로 눈으로 깨끗해지는 걸 보고, 그다음 **PSD(주파수별 세기)** — 이퀄라이저처럼 어느 주파수에 에너지가 얼마나 있는지. 두 관점이 같은 사실을 다른 각도로 말합니다: 느린 드리프트·전기잡음을 걷고 정답을 담은 운동 리듬(뮤·베타)만 남기니 필터된 게 정말 *깨끗*하다는 걸요."
**🖥️ 시연:** `python scripts/visualize.py 4 14` *(아무 subject/run이나 됨 — 평가관이 번호를 골라도 됨).*
**👉 시계열:** raw는 큰 느린 드리프트; 필터 후 0 주변 매끈; 색 띠=표시(파랑=T0, 주황=T1, 초록=T2); 동작 활동 보존.
**👉 PSD:** 필터 전=0–80 전체에 힘 + **60Hz 전기잡음** 뾰족; 후=7–30만 남고 30Hz 위 급강하, 봉우리 사라짐 — "깨끗"의 직접 증거.

**❓ Q&A**
- *그래프가 두 종류인데 왜 둘 다?* — 같은 신호 두 관점. 시계열=매끈해짐을 *눈으로*, PSD=어느 주파수가 남았는지 *증명*. 필터링은 하나, 그 결과를 두 각도로.
- *필터가 실제로 한 일?* — 원본은 ~0–80Hz가 섞임(80=160Hz 샘플링의 절반=Nyquist). 밴드패스가 **7–30Hz만 남기고 나머지를 거의 0으로** 누름.
- *전/후 둘 다 0–80인데 왜 다르게 보이나?* — x축은 일부러 같음; **내용**만 다름(전/후). 전체를 보여줘야 "대역 밖 제거"가 보임.
- *PSD 축은?* — x=주파수(Hz); y=세기, **dB**(로그) — 그 주파수가 얼마나 강한지.
- *왜 음수 dB?* — dB는 **기준(1 µV², 라벨 `re 1 µV²`) 대비**(로비 기준 층수처럼). 음수=기준보다 약함, 음수 세기 아님.
- *왜 7–30, 평가표의 40 아니고?* — 뮤·베타 ERD가 8–30에서 제일 강하고 30 위는 잡음↑; 8–30은 표준이고 0.658로 검증.
- *필터 종류?* — FIR(firwin), **zero-phase**(타이밍 안 밂); 수직 절벽이 아닌 가파른 비탈(유한 FIR).
- *average reference가 정확히?* — 매 순간 64채널 평균을 각 채널에서 빼기.
- *그 평균에 각 채널 고유 신호도 섞이지 않나?* — 섞임. 하지만 **공통은 평균에 살아남고 고유(부호 제각각)는 상쇄**돼서 평균≈공통; 빼도 각 채널 손실은 ~1/64(≈1.5%)뿐.
- *"기준"이 두 개 — 헷갈리지 말 것:* average reference는 *신호를 바꾸는* 전처리(`preprocessing.py`); 1 µV² 기준은 PSD의 dB 눈금일 뿐(MNE 그림, `viz.py`).
- *PSD는 어디서 계산?* — `viz.py`가 MNE `compute_psd(method="welch").plot()` 호출; 시각화일 뿐 모델엔 안 들어감 — 파이프라인은 필터된 *시계열* → 에폭 → CSP.
- *한 에폭 모양과 이유?* — `(64, 321)`: 64채널 × (2초 × 160Hz + 1)샘플, +1은 양 끝 포함.
- *T1/T2 뜻?* — 데이터셋 동작 표시; run 그룹에 따름(run 3/7/11 → 왼/오른 주먹; 5/9/13 → 양주먹/양발).
- *'영상'은?* — 주제·제출물에 영상 없음; 핵심은 "필터된 게 깨끗"이고 PSD가 보임.

→ **Yes**

---

## 2.2 Feature extraction
**📋 평가 기준 (평가표 원문 그대로):**
> *"Its nice to filter a signal, but it needs to mean something in the context of your data. Check that the significative frequencies for a motor imagery task are kept (~8-40Hz). If the program learns to select the relevant frequencies for classification its better, cf bonus questions."*
> (번역) 필터링은 좋지만 데이터 맥락에서 의미가 있어야 함. 운동상상의 유의미한 주파수(~8–40Hz)가 유지되는지 확인. 주파수를 스스로 선택 학습하면 더 좋음(보너스).

**📚 알아야 할 개념:**
- 이 항목은 **필터링이 *의미 있나*, 아무거나가 아닌가** — 운동상상 정보를 실제로 담은 주파수를 남겼나? 만 봅니다. (CSP 압축 얘기가 *아님* — 그건 Implementation.)
- **왜 뮤·베타(7–30Hz)가 의미 있는 대역인가:** 운동 영역엔 두 고유 리듬이 있어요 — **뮤(8–12Hz)**(쉴 때의 "공회전 소리")와 **베타(13–30Hz)**(운동을 다루는 리듬) — 둘 다 신경세포가 *동기화*돼 생기는 큰 출렁임. 손을 **움직이거나 상상하는** 순간 그 영역이 바빠져 동기화가 깨지고 뮤·베타 출렁임이 **작아짐**(= **ERD**), **반대쪽**에서(오른손 → 왼쪽 운동영역 C3 감소). 즉 **뮤·베타의 *크기* = "이 운동 영역이 켜졌나, 어느 쪽이냐"의 계기판** — 우리가 분류하려는 바로 그 정보.
- **왜 다른 데는 안 되나:** 7Hz 아래=느린 드리프트·눈 움직임; 30Hz 위=근육·50/60Hz 전기잡음 — 둘 다 운동 정보 없음. 정보가 8–30Hz에 몰려 있어 **7–30Hz가 그걸 살리고 나머지를 버림** — 그래서 의미 있는 필터링.

**🗣️ 할 말:** "이건 제 필터링이 *의미 있는지*, 아무 필터인지 묻는 항목입니다. 의미 있어요: **7–30Hz**, 바로 운동상상 ERD가 사는 뮤·베타를 남깁니다. 가령 50–60Hz를 남긴 필터는 깨끗해 보여도 운동 정보가 없죠. PSD에서 뮤·베타는 남고 나머지는 제거된 걸 봤습니다. (어느 세부 주파수가 중요한지 학습하는 건 FBCSP 보너스.)"
**🖥️ 시연:** `FMIN = 7.0, FMAX = 30.0`(`config.py`) + 뮤·베타 남은 필터 후 PSD.

**❓ Q&A**
- *왜 7–30Hz가 '의미 있고' 아무거나가 아닌가?* — 뮤(8–12)·베타(13–30)가 운동상상에 따라 변하는 리듬(ERD); 다른 대역은 그 정보를 안 담음.
- *왜 30까지만, 평가표의 40 아니고?* — ERD가 8–30에서 제일 강하고 30 위는 잡음↑; 8–30 표준, 0.658로 검증.
- *'의미 없는' 필터는 어떤 모습?* — 운동 정보 없는 대역(예 50–60Hz 전기잡음 구역)을 남기거나, 아예 필터를 안 하거나.
- *'적합한 주파수 학습'은?* — FBCSP 보너스(2.10)가 대역을 하위밴드로 쪼개 분류기가 가중.

→ **Yes**

---

## 2.3 Train
**📋 평가 기준 (평가표 원문 그대로):**
> *"The program has a train mode, sklearn score validation tools are used. The score for the training is displayed."*
> (번역) train 모드가 있고, sklearn 점수 검증 도구를 쓰며, 학습 점수가 표시됨.

**📚 알아야 할 개념:**
- **왜 학습 문제로 채점하면 안 되나?** 컨닝 — 외워버림(실측 1.0).
- **교차검증(`cross_val_score`)**: "20%(9문제) 숨기고 80%(36)로 학습, 숨긴 9개 채점"을 **10번**(`ShuffleSplit(10, test_size=0.2)`) 반복해 평균. **fold**=한 라운드. 숫자: 45=이 사람 문제 수, 9=20%, 36=80%, 10=`n_splits`; 각 fold는 k/9.
- 매 라운드 **새 모델을 만들고 버리는** 루프 — 한 번 학습이 아님.
- **누수**: CSP도 *배우므로* fold마다 학습분으로만 재학습해야 함. CSP를 **Pipeline 첫 단계**로 두면 해결(누수 vs 정상: 1.0 vs 0.844).
- **train의 독립 80/20 분할 둘:** **(A)** `ShuffleSplit×10` → 정직한 점수(게이트가 씀); **(B)** `train_test_split` 한 번 → 80%로 **배포 모델** 학습·저장 + 남은 20%는 predict용. A의 10개는 버림.

**🗣️ 할 말:** "`train`이 전체 파이프라인에 `cross_val_score`를 돌려 fold 점수와 평균을 출력하고, 모델을 학습·저장합니다."
**🖥️ 시연:** `python mybci.py 4 14 train`
```
[0.4444 1.0000 0.5556 0.6667 0.6667 1.0000 1.0000 0.6667 0.6667 0.6667]
cross_val_score: 0.7333
```
**👉 할 말:** "fold 정확도 10개(각 k/9라 들쭉날쭉), 평균 0.7333 = 학습 점수. 이후 80%로 저장 모델을 학습하고 남은 20%를 predict용으로 저장."

**❓ Q&A**
- *어떤 검증 도구?* — **전체 파이프라인**에 `cross_val_score` + `ShuffleSplit(n_splits=10, test_size=0.2)`.
- *45 / 36 / 9 / 10은 어디서?* — 45문제(3런×~15); 20%=9; 80%=36; 10=라운드 수.
- *"fold"가 뭔가?* — 한 번의 학습/시험 분할 + 그 점수; 10 fold 평균.
- *왜 10번, 1번이 아니라?* — 한 번 분할은 운에 좌우(0.44–1.0); 평균이라야 안정적.
- *한 번 학습하고 끝에 시험?* — 아님: 라운드마다 **새 CSP+LDA**를 36으로 학습, 9 채점, 버림.
- *그럼 저장 모델은?* — 별도 `train_test_split` 80% 학습·저장(+남은 20%) — `train`은 두 독립 작업(채점 A + 배포 모델 B).
- *왜 `KFold`가 아니라 `ShuffleSplit`?* — 독립 랜덤 80/20 반복; 둘 다 유효.
- *`fit`이 실제로 배우는 건?* — CSP는 `filters_`(4×64), LDA는 방향 `w`와 절편 `b`.
- *누수 없나?* — 없음: CSP가 첫 단계라 fold마다 학습분으로만 재학습.
- *왜 0.73인데 게이트는 60%?* — 60%는 **109명×6실험 평균(0.658)**, 한 사람 아님.
- *재현?* — `TPV_SEED=42`로 분할 고정; 미고정 시 약간 변동.

→ **Yes**

---

## 2.4 Predict
**📋 평가 기준 (평가표 원문 그대로):**
> *"There is a predict mode, which also uses validation tools. The prediction output is displayed (the id of the output class is enough)."*
> (번역) predict 모드가 있고 검증 도구도 사용, 예측 출력이 표시됨(클래스 id면 충분).

**📚 알아야 할 개념:**
- 저장된 `.joblib`엔 **파이프라인(80% 학습), held-out 20%(`X_test`,`y_test`), meta**가 들어 있음. `predict`가 그걸 불러와 **학습 때 못 본** 그 문제들을 재생하므로 정직한 정확도.
- **모델은 사람별**(뇌/전극이 사람마다 다름) → `predict 4 14`는 `train 4 14`가 먼저 있어야. 새 사람은 짧은 **캘리브레이션**(그 사람 데이터로 새 모델 학습)만 하면 됨.

**🗣️ 할 말:** "`predict`가 저장 모델을 불러와 held-out 20%(9문제)를 하나씩 재생하며 예측과 정답을 비교 — 재학습 없음."
**🖥️ 시연:** `python mybci.py 4 14 predict`
```
epoch 00:  [2]  [2] True
...
Accuracy: 0.6667
```
**👉 할 말:** "줄마다 **[예측] [정답] 일치**; id 1=상상 양손(T1), 2=상상 양발(T2). 이 9개는 학습 때 못 봄; 6/9=0.6667."

**❓ Q&A**
- *'검증 도구'는 어디?* — **held-out 검증셋**으로 돌리고 정답과 비교해 정확도.
- *왜 9문제만?* — `train`이 이 피험자/실험에서 빼둔 20%.
- *0.6667 vs train 0.7333?* — train=10-fold 평균, 이건=고정 9문제 한 번. 추정 방식 다름.
- *predict가 재학습?* — 아니요, `.joblib` 로드만; 추론=CSP 변환 + LDA 내적 한 번.
- *train과 같은 subject/run?* — 네; 산출물이 피험자/run별로 저장, 없으면 에러.
- *109명 밖 사람도 예측?* — 네, 짧은 캘리브레이션(그 사람 데이터로 새 모델). 109명은 고정 메뉴가 아님.
- *왜 사람별, 범용 아님?* — 뇌/전극이 달라 사람별(캘리브레이션) 모델이 BCI 표준이고 보통 더 나음. 학습이 빠르고 해석적(딥러닝 아님).

→ **Yes**

---

## 2.5 Realtime
**📋 평가 기준 (평가표 원문 그대로):**
> *"The prediction is made as the data is streamed to the processing pipeline. The program outputs the result between 0 and 2 seconds after the event was triggered."*
> (번역) 데이터가 스트리밍되는 동안 예측, 이벤트 후 0~2초 사이 결과 출력.

**📚 알아야 할 개념:**
- 파이프라인에 **한 이벤트(2초 에폭)씩** 흘려보냄(스트림처럼). 추론은 행렬곱+분산+내적 → **1밀리초 미만**. `assert latency < 2.0`이 예산 강제.
- "2초"는 대부분 이벤트 후 **2초 창 모으기**(TMIN=0…TMAX=2.0); 계산은 무시할 수준.

**🗣️ 할 말:** "predict가 2초 에폭을 하나씩 스트림처럼 처리; 이벤트당 지연 ~0.1–0.5 **밀리초** — 2초 예산의 1만분의 1 — 이고 `assert latency < 2.0`이 강제. 데이터가 사전 녹음이라 **시뮬레이션 스트림**(held-out 에폭을 하나씩 재생)이고, `mne-realtime` 대신 루프로 구현."
**🖥️ 시연:** 이벤트당 지연 `0.0001–0.0005초  OK (<2s)`.

**❓ Q&A**
- *진짜 라이브 스트림?* — 사전 녹음에 대한 이벤트 단위 시뮬레이션; 처리 모델(한 번에 하나, 2초 예산)은 라이브와 동일.
- *왜 빠른가?* — 반복 계산이 없음.
- *2초의 대부분은?* — 2초 창 모으기; 계산은 ~0.5ms.
- *2초는 어디서 보장?* — `predict.py`의 `assert latency < LATENCY_BUDGET_S`, + 에폭이 이벤트 후 2초 창.
- *왜 mne-realtime 안 씀?* — 루프로 "스트림, 2초 내"를 충족, 외부 실시간 의존성 불필요.

→ **Yes**

---

## 2.6 Integration
**📋 평가 기준 (평가표 원문 그대로):**
> *"Implementation was integrated to sklearn pipeline, inheriting from the baseEstimator and transformerMixin classes of sklearn."*
> (번역) 구현이 sklearn 파이프라인에 통합되고 BaseEstimator·TransformerMixin을 상속.

**📚 알아야 할 개념:**
- **sklearn** = 표준 부품 + 표준 배선 규격의 도구상자; **Pipeline** = 조립 라인(CSP 칸 → LDA 칸).
- **상속** = 자식 클래스가 부모 능력을 공짜로 물려받음. `class MyCSP(TransformerMixin, BaseEstimator)`가 두 규약을 상속:
  - **BaseEstimator** → `get_params`/`set_params` → **복제(clone)·튜닝·검사** 가능.
  - **TransformerMixin** → `fit_transform` 공짜 → 인정받는 **변환기**(`fit`/`transform`).
- 이게 `cross_val_score`가 **fold마다 복제·재학습**하게 해줌 — 누수 없는 채점의 토대지 형식이 아님.

**🗣️ 할 말:** "제 CSP가 `TransformerMixin`·`BaseEstimator`를 상속해 `fit`/`transform` 규약을 갖춰 `Pipeline([('csp', MyCSP()), ('clf', LDA())])`에 바로 들어갑니다. 그 상속이 `cross_val_score`가 fold마다 복제·재학습하게 하는, 정직한 채점의 토대예요."
**🖥️ 시연:** `isinstance(csp, BaseEstimator)=True`, `isinstance(csp, TransformerMixin)=True`, `get_params()={...}`, `fit_transform` 있음, `clone()` 동작, Pipeline에 꽂힘.

**❓ Q&A**
- *상속이 뭔가?* — 자식이 부모 메서드(여기선 sklearn 규약)를 안 다시 짜고 물려받음.
- *왜 두 base class 다?* — BaseEstimator=파라미터 관리(복제/튜닝/검사), TransformerMixin=변환기 규약. 둘 다라야 온전한 부품.
- *`clone`이 뭐고 왜 CV에 필요?* — 학습 안 된 추정기를 파라미터로 복사; `cross_val_score`가 fold마다 파이프라인을 clone해 학습분으로만 재학습(누수 없음).
- *왜 `__init__`에서 계산 안 함?* — sklearn 규약: 하이퍼파라미터 저장만, 학습은 `fit`; 아니면 `clone`이 깨짐.
- *변환기 vs 분류기?* — 변환기(`fit`/`transform`)는 데이터를 바꿈(CSP: 64×321→4); 분류기(`fit`/`predict`)는 라벨을 냄(LDA).
- *라이브러리 CSP 피함?* — 네; `mne.decoding.CSP`는 테스트 패리티용뿐.

→ **Yes**

---

## 2.7 Implementation — 차원 축소 (배점 큰 항목)
**📋 평가 기준 (평가표 원문 그대로):**
> *"A dimensionality reduction algorithm is implemented, the subject talks about PCA and CSP but other algorithms performing a dimensionnality reduction are feasible. Check that the student has a general understanding of the algorithm. It is allowed to use functions from libs like numpy or scipy for some tasks : the eigenvalues decomposition, singular values decompositon and covariance matrix estimation."*
> (번역) 차원 축소 알고리즘 구현(PCA/CSP 등) + 알고리즘 전반 이해. 고유값분해·SVD·공분산엔 numpy/scipy 허용.

**📚 알아야 할 개념 — CSP 3단계 (직접 구현 핵심, `csp.py`):**
목표: 한 문제(64 × 321 ≈ 2만 숫자)를 두 클래스가 갈리는 **4숫자**로.
1. **왜 줄이나.** 답은 출렁임 *크기*에만 있고 ~45문제로 2만차원에 경계를 못 그음(과적합). 4숫자면 깔끔한 경계.
2. **어떻게 섞나 — "레시피"와 `@`.** **공간 필터** = 64가중치. 행렬곱 `@` = "곱해서 더하기": `filters @ E`가 64채널을 한 **가상 채널**로 블렌딩. 축소 두 번: `64×321 →(@)→ 4×321 →(줄마다 분산)→ 4숫자`(채널 줄이고, 시간을 크기로 접음). 레시피 ①의 큰 가중치는 오른쪽 운동영역에 — "오른쪽 운동 감지기".
3. **필터를 어떻게 고르나 — 공분산 + 고유분해.** 클래스별 공분산 **C1, C2**(64×64). **일반화 고유문제 `eigh(C1, C1+C2)`**: 비율 `wᵀC1w / wᵀ(C1+C2)w`를 최대화하는 방향 `w`들 + 고유값 **λ = a/(a+b)**(한 클래스로 쏠린 비율). λ→1/0=감지기, **λ=0.5=쓸모없음**. **0.5에서 가장 먼 필터 4개** 보관.
4. **특징 = 로그-분산**: 가상 채널마다 분산 → 정규화 → log. (종 모양으로 펴고 선형화; 0.896 vs 0.884.)

**📚 4숫자가 들어가는 곳:** CSP의 4숫자는 **LDA 분류기**로 들어감(두 클래스 사이에 선을 긋는 것; LDA는 보너스 2.11-D로 직접 구현). LDA는 *차원 축소가 아님* — 이 항목은 축소기(CSP)만 다룸.

**🗣️ 할 말:** "`csp.py`에 **CSP를 밑바닥부터** 구현 — import는 numpy, `scipy.linalg.eigh`(허용), `sklearn.base`, 자작 `generalized_eigh`뿐; 라이브러리 CSP 없음. 4단계: 답이 출렁임 크기고 45문제로 2만차원 못 채우니 줄이고; 클래스별 공분산; `eigh(C1,C1+C2)`로 고유값이 0.5에서 가장 먼 필터를 고르고; `@`로 섞어 로그-분산 → 4숫자."
**🖥️ 시연:** `pytest tests/test_csp_parity.py -v` → **PASSED**(같은 분할에서 `mne.decoding.CSP`와 0.05 이내, 단일 피험자 ≥ 0.60).
**🗣️ 코드 투어:** `_class_cov` = `E@E.T` → trace 정규화(센 문제가 지배 못하게) → 평균 → 대칭화 → **shrinkage**(`reg=0.01`, `eigh` 안정). `fit` = C1,C2 → `eigh` → `|λ-0.5|` 정렬 → `filters_`(상위 4×64). `transform` = `filters_ @ E` → 분산 → log.

**❓ Q&A**
- *여기서 고유값은?* — 그 필터 출렁임이 한 클래스로 쏠린 비율; 0.5에서 멀수록 판별력↑.
- *`@`가 뭔가?* — 행렬곱 = 곱해서 더하기; 64채널을 한 가상 채널로 블렌딩.
- *공분산은?* — 클래스 안 채널들이 같이 출렁이는 64×64 표; CSP 재료.
- *왜 일반화 고유문제(두 행렬)?* — 비율 `wᵀC1w/wᵀ(C1+C2)w` 최대화라 분모에 (C1+C2).
- *왜 CSP가 PCA보다?* — PCA=비지도(큰 변화, 잡음에 낚임), CSP=지도(클래스 차이).
- *왜 분산이 아니라 로그-분산?* — 종 모양으로 펴고 곱셈 차이를 선형화.
- *왜 trace 정규화/shrinkage?* — 센 문제가 지배 안 하게; shrinkage로 공분산을 잘 조건화해 `eigh` 안정.
- *`filters_` vs `patterns_`?* — filters_=신호→특징(분석), patterns_=특징→두피(시각화).
- *왜 정확히 2클래스?* — CSP는 2클래스용; 우리 6실험이 전부 2지선다라 OvR 불필요.
- *그럴듯한 게 아니라 맞다고 증명?* — MNE와 패리티(<0.05); 보너스 Jacobi가 scipy와 ~1e-14.
- *금지 함수 썼나?* — 아니요; 기준이 고유값분해/SVD/공분산에 numpy/scipy 허용, 그것조차 직접 재구현.

→ **Yes**

---

## 2.8 Score (60% 게이트)
**📋 평가 기준 (평가표 원문 그대로):**
> *"There has to be a script executing training over each subject and computing the mean of scores over each subjects, by type of experiment runs. The mean of the resulting six means (corresponding to the six types of experiment runs) has to be superior or equal to 60%."*
> *(등급)* *"Over 60% add a point for every 1%."* — *"Rate it from 0 (failed) through 5 (excellent)"*
> (번역) 각 피험자 학습 + 실험 유형별 점수 평균 스크립트, 6개 평균의 평균 ≥ 60%. (등급) 60% 위 1%마다 1점, 0~5.

**📚 알아야 할 개념:**
- **6개 실험**(각 2지선다): 0 실제 손좌우 · 1 상상 손좌우 · 2 실제 양손/발 · 3 상상 양손/발 · 4 실제vs상상(손) · 5 실제vs상상(발).
- 게이트 = 실험마다 **109명** 교차검증 정확도 평균(6개) → 그 **6개의 평균** ≥ 0.60. 사람별 채점은 2.3의 누수 없는 `cross_val_score`.

**🗣️ 할 말:** "`python mybci.py`(인자 없음)가 109명×6실험을 교차검증, 실험별 평균(6개) → 그 6개의 평균. 전체 = **0.658 ≥ 0.60 → 통과**; 65.8%라 등급 상한 **5/5**."
**🖥️ 시연:** `python scripts/validate_60.py` (디펜스엔 **전체** 실행; `--fast 5`는 미리보기).

**❓ Q&A**
- *60% 계산?* — 실험별 평균(6개) → 그 6개의 평균, 한 번에 평탄 평균 아님.
- *왜 `--fast 5`는 ~0.71인데 전체는 0.658?* — 앞 5명이 쉬움; 전체 109엔 어려운 분도. 공식=전체.
- *피험자 실패하면?* — `run_all`은 데이터/IO 오류만 좁게 skip; 진짜 버그는 터뜨림. 그 피험자 `train`을 단독 재실행 가능.
- *게이트 seed?* — 미고정 시 0.65 부근 변동; 기준은 grand mean.
- *왜 사람마다 편차?* — BCI는 사람 편차가 유명; 그래서 109명 평균 + 사람마다 10-fold 평균.
- *왜 exp0(왼손vs오른손)이 제일 약하나(~0.57)?* — 둘 다 손이고 인접·대칭 운동영역이라 반대쪽 차이가 작음; 양손vs양발(측면 vs 정중앙)은 훨씬 쉬움(~0.71–0.84).
- *시간?* — 캐시 시 ~2.5분; 최초엔 ~3.1GB 다운로드.

→ **Yes / 5/5**

---

## 2.9 Bonus · Datasets
**📋 평가 기준 (평가표 원문 그대로):**
> *"Are there other datasets processed by the program ? Is the scoring on those datasets correct ? Try to assert this taking into account the noise and the general quality of the dataset compared to the one given in the subject."*
> (번역) 다른 데이터셋도 처리? 채점이 올바른가(주제 대비 노이즈·품질 고려)?

**📚 알아야 할 개념:**
- **BCI Competition IV-2a**(moabb): 구조가 다른 데이터셋 — **22채널, 250Hz, 288 trial**. 파이프라인이 **그대로** 도는 건 **MyCSP가 채널 수 무관**(공분산이 채널수²)이고 `external.py`가 7–30Hz를 moabb에 넘기기 때문.
- 프로젝트 `mne_data/`로 받음(호출 동안만 moabb의 `MNE_DATASETS_BNCI_PATH`를 프로젝트로, 끝나면 복원) + 필수 게이트와 **별도 실행**.

**🗣️ 할 말:** "다른 데이터셋 BCI IV-2a — 22채널·250Hz — 를 붙였고 파이프라인이 그대로 **0.855**. 0.658보다 높은 건 IV-2a가 정제된 대회 데이터라 노이즈가 적어서 — 평가표가 말한 품질 고려 그대로."
**🖥️ 시연:** `python scripts/bonus_demo.py` → `[G] 2nd dataset BCI IV-2a (subj 1) : 0.8552`(모양 288×22×1001).

**❓ Q&A**
- *64채널 코드가 왜 22채널에서?* — 채널 수 가정이 코드에 없음; 공분산·고유문제가 입력에 맞춤.
- *0.855 > 0.658이 의심?* — 아님: 깨끗한 데이터 → 높은 점수; 기준도 품질 고려 요구.
- *전처리 같나?* — 네, moabb 패러다임으로 7–30Hz; 같은 왼손vs오른손 상상.
- *데이터 어디 저장?* — 프로젝트 `./mne_data/MNE-bnci-data/`(~83MB), gitignore라 폴더와 함께 이동.
- *게이트와 합쳐지나?* — 아니요, 별도 실행; 60% 게이트는 PhysioNet만.

→ **Yes**

---

## 2.10 Bonus · Feature engineering
**📋 평가 기준 (평가표 원문 그대로):**
> *"Try to evaluate the relevance of the preprocessing stage and how are the data feeded to the algorithm. The use of fourier or wavelet transform, and anything that transform the data before the processing is a plus."*
> (번역) 전처리 적절성 + 데이터 입력 방식 평가. 푸리에/웨이블릿 등 처리 전 변환은 가산점.

**📚 알아야 할 개념:**
- **FBCSP(Filter-Bank CSP)**가 7–30Hz를 **4개 하위 밴드(8-12 / 12-16 / 16-20 / 20-30)**로 쪼개 밴드마다 CSP → 8특징, 이어붙임.
- *왜 쪼개나:* 단일 7–30은 뮤·저베타·고베타를 뭉침; **ERD 세기·분포가 주파수마다 달라** FBCSP가 밴드별 공간패턴을 학습하고 분류기가 밴드 중요도를 가중 — 단일 CSP에 없는 주파수 해상도. 푸리에/웨이블릿 계열의 특징공학. 밴드패스가 `fit`/`transform` 안이라 CV가 fold마다 재학습(누수 없음).

**🗣️ 할 말:** "주파수 변환 두 층: 필수 FIR 밴드패스가 하나; 보너스 **FBCSP**가 대역을 4 하위밴드로 쪼개 밴드마다 CSP라 분류기가 주파수 중요도를 가중. 솔직히 여기선 0.822 vs 0.844 — 8특징이 ~45문제에 약간 과적합 — 이지만 엄밀히 더 표현력 있고 견고한 특징공학 시연."
**🖥️ 시연:** `[F] Filter-Bank CSP (4 sub-bands) : 0.8222`(기본 CSP 0.8444).

**❓ Q&A**
- *FBCSP가 CSP보다 더 하는 일?* — 밴드별 공간패턴, 분류기가 밴드 중요도 학습.
- *왜 항상 더 정확하지 않나?* — 특징 증가 → ~45문제에 약간 과적합; 보장된 승리가 아닌 정교함 시연.
- *데이터 입력 방식?* — 각 하위밴드 → CSP → 로그-분산 → 이어붙여(8숫자) 분류기에.

→ **Yes**

---

## 2.11 Bonus · Implementations
**📋 평가 기준 (평가표 원문 그대로):**
> *"How deep did the student dig into his implementation ? ( Did he implement his own eigenvalues decomposition, SVD, or covariance matrix estimation ? ) ( Did he implement a complex dimensionality reduction algorithm ? ) Is there some kind of hyperparameter tuning or learning ? Did he implement his own classifier ?"*
> (번역) 얼마나 깊이 — 자작 고유값분해/SVD/공분산? 복잡한 차원축소? 튜닝/학습? 자작 분류기?

**📚 개념 + 🖥️ 시연 (하위 항목별):**

**A. 자작 고유분해(`jacobi.py`).** cyclic Jacobi가 비대각 원소를 한 쌍씩 회전으로 0으로, 대각이 될 때까지 sweep — 대각=고유값, 누적 회전=고유벡터. **일반화** `C1 w = λ(C1+C2)w`는 **백색화**로 표준화 후 Jacobi 두 번. `+,*,sqrt,sign,matmul`만 — `eig/eigh/svd`·scipy 없음.
```
[A] Jacobi 정확도 : 0.8444   표준 고유분해 vs numpy : 5.33e-14   일반화 vs scipy : 1.55e-15
```
**C. 하이퍼파라미터 튜닝(`evaluate.tune`).** `GridSearchCV`로 `n_components ∈ {4,6,8}`, inner fold마다 CSP 재학습 = **누수 없는 중첩 CV**. 공식 게이트는 비교 공정성 위해 4 고정.
```
[C] tuned {'csp__n_components': 8} -> cv 0.8444
```
**D. 자작 분류기(`own_lda.py`).** 풀드 공분산, **`w = Σ⁻¹(μ1−μ0)`**, 사전확률 보정 절편, 판정 = sign(`w·x+b`), `np.linalg.solve`만.
```
[D] OwnLDA 정확도 : 0.7889   sklearn LDA와 일치율 : 100.0%
```
(+ **F. FBCSP**도 "복잡한 차원축소"에 해당.)

**🗣️ 할 말:** "밑바닥까지 팠습니다: 자작 고유솔버(Jacobi, scipy와 ~1e-14), 누수 없는 하이퍼파라미터 튜닝, 자작 LDA(sklearn과 100% 일치) — 평가표 요구를 넘어서요."

**❓ Q&A**
- *정말 고유분해 라이브러리 없이?* — `jacobi.py`는 기본 연산만; scipy와 ~5e-14.
- *Jacobi 작동 방식?* — 회전으로 비대각을 0으로, 대각=고유값; 백색화로 일반화 문제 축소 후 Jacobi 두 번.
- *튜닝 누수 없나?* — 없음; inner fold마다 CSP 재학습.
- *왜 게이트는 4성분 고정?* — 공정·비교 가능한 채점; 튜닝은 데모.
- *OwnLDA가 sklearn과 같나?* — 예측 100% 일치(같은 풀드 공분산 + Σ⁻¹(μ1−μ0)).

→ **Yes**

---

## 2.12 Ratings & Conclusion
**📋 원문:** *"Don't forget to check the flag corresponding to the defense"* — *"Ok / Outstanding project"*; 플래그 *"Empty work / Incomplete work / Cheat / Crash / Forbidden function"*; *"Leave a comment on this evaluation ( 2048 chars max )"*.

**🗣️ 할 말:** "부정 플래그 하나도 해당 없음 — 특히 **Forbidden function**: 기준이 고유값분해/SVD/공분산에 numpy/scipy를 허용했고 그것조차 재구현했습니다. 필수 0.658 통과 + 보너스 5개 전부 구현·검증 → **Outstanding project** 제안."

**마무리 한 문장:** "PhysioNet 64채널을 7–30Hz로 걸러 ERD를 드러내고, 직접 구현한 CSP로 4숫자로 압축, LDA로 분류, 누수 없는 교차검증으로 109명×6실험 평균 **0.658(≥60%)** 달성; 자작 고유분해 포함 보너스 5개 전부 구현."

→ **Outstanding** ⭐

---

# 3. 명령어 런북 (순서대로)

```bash
source .venv/bin/activate
python scripts/validate_60.py              # 전체 60% 게이트 → 0.658  (디펜스 전에도)
pytest -m "not network"                    # 빠른 단위 테스트
pytest tests/test_csp_parity.py -v         # 자작 CSP vs mne, 0.05 이내
python scripts/visualize.py 4 14           # raw vs 7–30Hz + PSD
python mybci.py 4 14 train                 # cross_val_score + 모델 저장
python mybci.py 4 14 predict               # held-out 스트림
python mybci.py                            # 6실험 평균 → 0.658
python scripts/bonus_demo.py               # 보너스 A,C,D,F,G 한 번에
```

---

# 4. 최종 체크리스트

| 평가표 항목 | 상태 | 근거(실측) |
|---|---|---|
| Preprocessing (plot) | ✅ | raw vs 7–30Hz + PSD; 60Hz/드리프트 제거 |
| Feature extraction | ✅ | 뮤·베타(7–30Hz) 유지 |
| Train | ✅ | `cross_val_score` 0.7333 표시 |
| Predict | ✅ | held-out 6/9 = 0.667 출력 |
| Realtime | ✅ | 이벤트당 ~0.0005초 < 2초 |
| Integration | ✅ | BaseEstimator + TransformerMixin; clone 동작 |
| Implementation (차원축소) | ✅ | 자작 CSP; 패리티 PASS(<0.05) |
| Score | ✅ **5/5** | 0.658 ≥ 0.60 |
| Bonus: Datasets | ✅ | BCI IV-2a 0.855 (22ch/250Hz) |
| Bonus: Feature engineering | ✅ | FBCSP 0.822 (4 하위밴드) |
| Bonus: Implementations | ✅ | Jacobi 5e-14 · 튜닝 · OwnLDA 100% |
| Ratings | ⭐ Outstanding | 부정 플래그 없음 |

> **위 각 항목이 자기완결적입니다 — 개념·기준·시연·Q&A가 한 곳에. 위에서 아래로 걸으면 프로젝트 전체를 디펜스할 수 있어요.**

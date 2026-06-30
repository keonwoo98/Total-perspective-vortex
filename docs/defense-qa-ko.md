# Total Perspective Vortex — 디펜스 Q&A (한국어판)

> 영어판 [`defense-qa-en.md`](defense-qa-en.md)와 **구조·내용이 똑같고 언어만 한국어**인 문서입니다.
> 항목마다 **📋 무엇을 보나 → 🖥️ 무엇을 시연하나 → 🗣️ 무슨 말을 하나 → ❓ 깊은 Q&A 뱅크**(기본 질문 + 까다로운 질문) 순서입니다.
> **2부**는 개념 복습이라 어떤 추가 질문도 답할 수 있게 해줍니다. **6부**는 교차 난문 모음입니다.
> 실제 평가는 영어로 진행되므로, **📋 평가 기준은 평가표 영어 원문 그대로** 두고 바로 아래에 한국어 번역을 붙였습니다. 나머지는 모두 쉬운 한국어예요.
> 실측 숫자는 전부 이 코드베이스(PhysioNet eegmmidb, MNE + scikit-learn)에서 직접 측정한 값입니다.

---

## 이 문서 사용법

1. **1부**(피치 + 지도)를 훑고 60초 피치를 연습하세요.
2. **2부**(개념)를 각 아이디어가 또렷해질 때까지 읽으세요 — 이게 Q&A에서 안 흔들리게 해줍니다.
3. 디펜스 중엔 **3부**를 위에서 아래로 따라가세요(평가표 순서 그대로).
4. **6부**는 보험입니다 — 제일 어려운 질문과 그 답.

표기: 🗣️ = 이렇게 말하기, 🖥️ = 실행/보여주기, 👉 = 화면 가리키기, ❓ = 질문 + 답.

---

# 0. 디펜스 전 준비 (한 번만)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 데이터셋을 프로젝트 폴더로 받고 60% 게이트 확인. 최초엔 느림.
python scripts/validate_60.py            # 전체 109명 → 평균 0.658
python scripts/bonus_demo.py             # 보너스 데이터셋(BCI IV-2a)도 ./mne_data로 받아옴
```

- 두 데이터셋 모두 **프로젝트 안 `./mne_data/`**에 있습니다: PhysioNet(~3.1GB)과 BCI IV-2a(~83MB). 폴더와 함께 이동합니다. `./mne_data/`는 gitignore라 코드만 커밋됩니다.
- 평가 컴퓨터에선 위 두 명령을 미리 한 번 돌려두세요(최초 다운로드는 인터넷 필요).

---

# 1. 60초 피치 + 프로젝트 지도

**피치(이렇게 말하세요):**
> "이건 EEG 뇌-컴퓨터 인터페이스입니다. PhysioNet의 운동/운동상상 녹음(109명, 64채널)에서, 사람이 어떤 동작을 하거나 상상했는지를 **뇌 신호만으로** 분류합니다 — 왼손 vs 오른손, 양손 vs 양발, 실제 vs 상상. 파이프라인은 3단계입니다: **(1) 전처리** — 평균 기준, 7–30Hz 밴드패스, 2초 에폭; **(2) 차원 축소** — 64채널을 판별력 있는 4숫자로 바꾸는 **직접 구현한 CSP**; **(3) 분류** — LDA. scikit-learn의 `cross_val_score`로 정직하게 채점합니다. 필수 요구는 **직접 구현한 차원 축소 알고리즘을 sklearn에 통합**하는 것과, 6가지 실험 유형을 109명 평균내어 **정확도 ≥ 60%**입니다. 저는 **0.658**을 달성했고, 보너스 5개를 전부 구현했습니다."

**지도:**
```
1 EEG 표(64채널×160Hz) → 2 목표(6개 2지선다, ≥60%) → 3 채널(C3/C4) →
4 에폭(≈45문제) → 5 ERD(반대쪽 휴식리듬 감소) → 6 필터(7–30Hz) →
7 CSP(64×321 → 4숫자) → 8 LDA(4숫자 → 판정) → 9 교차검증 + 누수방지 + 60% 게이트
```

---

# 2. 개념 복습 (이걸로 무엇이든 답할 수 있게)

### 2.1 EEG = 머리에 붙인 마이크 64개
전극 64개가 각자 **초당 160번** 전압을 재요(단위 **마이크로볼트 µV**). 전극 하나 = 출렁이는 선 한 줄; 64개가 모이면 **(64 × 시간)** 표. 약 125초 녹음은 `125초 × 160Hz = 20000` 샘플이라 **(64, 20000)**.

### 2.2 채널과 C3/C4 (받아들일 사실)
전극 이름이 머리 위치를 뜻합니다: 앞글자=구역(`C`=정수리/**운동 영역**), 숫자=좌우(**홀수=왼쪽, 짝수=오른쪽, z=정중앙**). 그래서 **C3**(왼쪽 운동영역) ↔ 오른손, **C4**(오른쪽 운동영역) ↔ 왼손 — 뇌는 **반대쪽**을 제어합니다. `C*` 7개 채널이 손/발 신호가 제일 강한 곳.
**"그냥 C3/C4만 쓰면?"** 6개 과제마다 신호가 숨은 채널이 달라요(손 → C3/C4, 발 → Cz). 실측(10명 평균): C3 혼자 ≈ 0.48, C3+C4 ≈ 0.57, 하지만 **양손 vs 양발**에선 64채널+CSP = **0.71** vs 2채널 0.56. 그래서 채널을 손으로 안 고르고 — **CSP가 과제마다 최적 채널 조합을 자동으로** 찾습니다.

### 2.3 목표 = 6개의 **2지선다** 게임, 평균 ≥ 60%
각 과제는 둘 중 하나 고르기이고, *무엇과 무엇*만 바뀝니다:

| 실험 | runs | 무엇 vs 무엇 |
|---|---|---|
| 0 | 3,7,11 | 실제 **왼손 vs 오른손** |
| 1 | 4,8,12 | 상상 **왼손 vs 오른손** |
| 2 | 5,9,13 | 실제 **양손 vs 양발** |
| 3 | 6,10,14 | 상상 **양손 vs 양발** |
| 4 | 3,7,11 vs 4,8,12 | **실제 vs 상상**(손) |
| 5 | 5,9,13 vs 6,10,14 | **실제 vs 상상**(양손/발) |

점수 = 실험마다 109명 평균(6개) → **그 6개의 평균 ≥ 0.60**. 우리는 **0.658**.

### 2.4 에폭, 그리고 "321"의 정체
데이터에 **표시(annotation)**가 박혀 있어요: `T0`=휴식, `T1`/`T2`=두 동작(뜻은 run 그룹마다 정해짐). 휴식은 버리고 **T1/T2마다 2초 창을 오려냄** → 한 문제 = **(64채널 × 321샘플)**.
**왜 321 = 2×160 + 1?** 울타리 기둥 세기: 0초~2.0초를 양 끝 포함해 찍으면 **간격은 320개지만 점은 321개**(시작점 0초가 +1). 한 사람·한 실험 ≈ **45문제**(3런 × ~15), 라벨은 거의 반반(예: 23/22).

### 2.5 ERD — 이게 가능한 이유
쉬는 운동 영역은 8–30Hz로 크게 출렁여요(신경세포가 **장단 맞춰** 박수치듯). 그 영역을 **쓰면**(움직이거나 상상하면) 박자가 깨져 출렁임이 **줄어듭니다**. 이게 **ERD(Event-Related Desynchronization)**이고, **반대쪽**에서 일어나요: 오른손 → C3 조용; 왼손 → C4 조용.
실측(20명 평균, µV²): 왼손 C3=83.1 / C4=**78.7**; 오른손 C3=**82.2** / C4=89.3 — 반대쪽 패턴이 진짜지만 **미묘**합니다. 그래서 CSP+LDA로 키워야 하죠. **핵심: 답은 신호의 정확한 모양이 아니라 출렁임의 *크기*(분산)에 있어요.**

### 2.6 필터 = 평균 기준 + 7–30Hz FIR(firwin)
1. **평균 기준**: 매 순간 64채널 평균을 빼서 모든 전극에 공통인 신호(기준점·전역 잡음)를 제거 → 각 채널 고유 활동만. (교실 웅성거림을 빼면 각자 목소리가 들리듯.)
2. **7–30Hz 밴드패스**: ERD는 **뮤(8–12Hz) + 베타(13–30Hz)**에 삶. 7Hz 아래=느린 드리프트, 30Hz 위=근육/50–60Hz 전기잡음 → 둘 다 버림.
3. **firwin** = **FIR** 필터를 만드는 레시피(각 출력 = 이웃들의 가중평균). **위상을 안 밀고**(타이밍 보존) 항상 안정적. 경계는 칼절벽이 아니라 가파른 비탈.
4. **긴 연속 신호를 먼저 필터링**한 뒤 에폭으로 자릅니다(2초 조각에 걸면 가장자리가 망가짐).
실측(C3 대역별 힘, 전→후): 0–7Hz **3743 → 8**, 7–30Hz **913 → 164**, 30–80Hz **162 → 7**. 4배 큰 드리프트에 파묻혔던 ERD 대역이 필터 후 우뚝 남습니다.

### 2.7 CSP 3단계 (직접 구현 핵심: `tpv/csp.py`)
**목표:** 한 문제(64×321 = 약 2만 숫자)를 두 클래스가 갈리는 **4숫자**로.

**(1) 왜 줄이나.** 답은 출렁임 *크기*에만 있고, 45문제로는 2만 차원에 경계를 못 그어요(과적합). 4숫자면 깔끔한 경계가 가능.

**(2) 어떻게 섞나 — "레시피"와 `@`.** **공간 필터** = 채널마다 가중치, 64개. 한 문제에 행렬곱 `@`로 적용 = "**곱해서 더하기**": `filters @ E`가 64채널을 한 **가상 채널**로 블렌딩. 축소가 두 번: `64×321 →(@)→ 4×321 →(줄마다 분산)→ 4숫자` — 먼저 채널을 줄이고, 그다음 시간을 출렁임 크기로 접음. (레시피 ①의 큰 가중치는 FC4/C4/C6/FC6/CP4 — 전부 **오른쪽 운동영역** — 즉 "오른쪽 운동영역 감지기".)

**(3) 필터를 어떻게 고르나 — 공분산 + 고유분해.** 클래스별 **공분산** C1, C2(64×64, "채널이 같이 출렁이는 정도")를 만듦. **일반화 고유문제 `eigh(C1, C1+C2)`**: 비율 `wᵀC1w / wᵀ(C1+C2)w`를 최대화하는 방향 `w`들과 고유값 **λ = a/(a+b)**(그 필터 출렁임 중 한 클래스 비율)를 한 번에. λ→1=한쪽 감지기, λ→0=반대쪽, **λ=0.5=쓸모없음**. **λ가 0.5에서 가장 먼 필터 4개**를 보관. (실측: λ=0.378, 0.619로 갈리고 `a/(a+b)`가 λ과 정확히 같음.)

**(4) 특징 = 로그-분산.** 가상 채널마다: 분산 → 비율로 정규화 → **log**. log를 쓰는 건 분산이 한쪽으로 쏠려서(log → 종 모양, LDA에 좋음) + 곱셈 차이를 덧셈 차이로 바꿔(직선 경계에 좋음). 실측: log 0.896 vs 미적용 0.884.

**용어(같은 것의 여러 이름):** *레시피 = 공간필터 = 고유벡터 = `filters_` 한 줄 = 64가중치* ↔ *점수 = λ = 고유값 = a/(a+b)*. 이름 앞 `_`(`_class_cov`)=내부 도우미; 뒤 `_`(`filters_`)=`fit`에서 학습된 값(sklearn 규칙).
**왜 PCA가 아니라 CSP?** PCA는 비지도(가장 큰 변화, 큰 잡음에 낚임); CSP는 **지도**(클래스 간 차이가 가장 큰 방향). 우리 일은 '구별'이라 CSP.

### 2.8 LDA = 울타리 긋기 (보너스는 `tpv/own_lda.py`)
CSP의 4숫자 = **4차원 점**; 두 클래스가 두 무리를 이룸. 선이 **둘** 있어요: **방향 `w`**(무리를 *관통*하는 화살표, 중심→중심)와 **울타리**(`w`에 수직, 무리 *사이*). 우리는 `w`를 계산하고, 울타리는 따라옵니다.
수학: 중심 `μ0, μ1`; **`w = Σ⁻¹(μ1 − μ0)`**(중심→중심 방향을 퍼짐 `Σ`로 보정 → 노이즈 큰 축은 덜 믿음); 판정 = **`w·x + b`**의 부호. LDA도 4→1로 줄이지만(`w`에 투영) 그 뒤 눈금과 비교해 클래스를 출력하므로 *분류기*.
실측(exp2): μ0=[-1.93,-1.78,-1.07,-1.13], μ1=[-1.11,-1.22,-1.79,-1.77], w=[12.92,7.86,-10.4,-2.13], b=13.46 → 양손 전부 음수, 양발 전부 양수 = 100% 분리.

### 2.9 교차검증, 누수, A/B 분할
**컨닝:** 배운 문제로 채점하면 부풀려짐(실측 1.0).
**`cross_val_score`** = "20%(9문제) 숨기고 80%(36문제)로 학습, 숨긴 9개로 채점"을 **10번** 반복(`ShuffleSplit(10, test_size=0.2)`)하고 평균. 매 라운드 **새 모델을 만들고 버리는** 루프이고, 수동 루프가 이를 정확히 재현. 실측(피험자4, exp3): folds `[0.44,1,0.56,…]`, 평균 **0.7333**(각 fold는 k/9).
**누수:** CSP도 *배우므로*, CSP가 필터 만들 때 시험 문제를 보면 숨은 컨닝. 해결: CSP를 **Pipeline 첫 단계**로 → `cross_val_score`가 **fold마다 학습분으로만 CSP 재학습**. 실측: 누수 1.0 vs 정상 0.844(**0.156** 부풀림).
**`train`의 독립적인 80/20 분할 둘:** **(A)** 전체에 `ShuffleSplit×10` → **정직한 점수**(60% 게이트가 쓰는 값); **(B)** `train_test_split` 한 번 → 80%로 **배포 모델**을 학습해 저장 + 남은 20%는 `predict`용. A의 10개 모델은 버리고, B가 굴릴 모델. A가 필요한 건 한 번 분할은 운에 좌우돼서, 10번 평균을 내기 위함.

### 2.10 sklearn 통합 = "표준 플러그"
sklearn = 표준 부품과 표준 배선 규격을 가진 도구 상자; **Pipeline** = 조립 라인(CSP 칸 → LDA 칸). **상속** = 자식 클래스가 부모 능력을 공짜로 물려받음. `class MyCSP(TransformerMixin, BaseEstimator)`:
- **BaseEstimator** → `get_params`/`set_params` → sklearn이 **복제·튜닝·검사** 가능.
- **TransformerMixin** → `fit_transform` 공짜 → 인정받는 **변환기**.
덕분에 Pipeline 콘센트에 꽂히고 `cross_val_score`가 **fold마다 복제·재학습** 가능 — 바로 이게 누수 없는 채점을 가능하게 합니다. 형식이 아니라 정직한 평가의 토대.

### 2.11 사람별 모델 (하나의 범용 신경망 아님)
모델은 **사람마다** 학습 — "*무슨 동작*"을 맞히지 "누구"가 아님. 뇌/전극이 사람마다 달라 사람별(캘리브레이션) 모델이 BCI 표준이고 보통 범용보다 낫습니다. 학습은 **빠르고 해석적**(고유분해 한 번 + 닫힌 형태 LDA), 몇 시간 도는 딥러닝과 다름. 60% 게이트는 **사람마다 모델 하나**를 학습해 평균.

---

# 3. 평가표 — 항목별

## 3.1 Preprocessing — *"Watch it for the plot"*
**📋 평가 기준 (평가표 원문 그대로):**
> *"Check if the data were parsed then visualized with a script, showing raw and filtered data. The plots should look like what is shown in the video, the filtered signal being "cleaner"."*
> (번역) 데이터를 파싱한 뒤 스크립트로 시각화해 raw와 filtered를 보여주는지, 필터된 신호가 더 "깨끗"한지 확인.

**🗣️ 말하기:** "전처리부터 보여드리겠습니다 — 원본 녹음을 분석에 쓸 수 있는 형태로 바꾸는 단계입니다. 스크립트 하나로 한 피험자의 EEG를 불러와 **두 가지 관점에서, 각각 필터 전/후로** 보여줍니다. 먼저 **시계열** — 원본 출렁이는 신호와 7–30Hz 필터된 신호를 나란히 — 눈으로 직접 깨끗해지는 걸 보실 수 있습니다. 그다음 **PSD(주파수별 세기)** — 일종의 이퀄라이저처럼, 어느 주파수에 에너지가 얼마나 있는지를 보여줍니다. 이 두 관점이 같은 사실을 다른 각도에서 말해줍니다: 느린 드리프트와 전기 잡음을 걷어내고 **정답을 담은 운동 리듬(뮤·베타)만** 남기니, 필터된 신호가 정말로 더 *깨끗*하다는 거죠."
**🖥️ 시연:** `python scripts/visualize.py 4 14` *(아무 subject/run이나 됨 — 체리피킹 아님을 보이려 평가관이 번호를 골라도 됨.)*
**👉 시계열:** "raw는 느린 베이스라인 드리프트가 크고, 필터 후엔 0 주변에서 매끈합니다. 색칠된 구간은 표시(파랑=T0 휴식, 주황=T1, 초록=T2)이고 동작 관련 활동은 보존됩니다."
**👉 PSD:** "필터 전: 저주파에 힘이 몰리고 **60Hz 전기잡음** 뾰족, 80Hz까지 신호. 필터 후: **7–30Hz만 남고** 30Hz 위 급강하, 60Hz 사라짐 — 이게 '깨끗'의 직접 증거."

**❓ Q&A**
- *왜 7–30Hz?* — 운동상상 뮤·베타 리듬(ERD)이 거기 살고, 평가표도 ~8–40Hz 유지를 요구.
- *필터 종류?* — **FIR**(firwin 설계); **zero-phase**라 이벤트 타이밍을 안 밀어요.
- *왜 30Hz에서 수직이 아니라 비스듬?* — 유한 FIR이라 전이 대역이 가파르되 유한; 7/30Hz 바로 밖은 점점 줄어들 뿐 0이 아님.
- *평균 기준은 왜?* — 모든 전극 공통 활동(기준점·전역 잡음)을 빼서 각 채널의 국소 활동만 남김.
- *'영상'은?* — 주제·제출물에 영상 없음; 핵심 요구는 '필터된 게 깨끗'이고 PSD가 명확히 보임.
- *색칠 색은?* — MNE 자동 배정; 여기선 파랑=T0, 주황=T1, 초록=T2(범례 확인, 색 자체에 의미 없음).

→ **Yes**

## 3.2 Feature extraction
**📋 평가 기준 (평가표 원문 그대로):**
> *"Its nice to filter a signal, but it needs to mean something in the context of your data. Check that the significative frequencies for a motor imagery task are kept (~8-40Hz). If the program learns to select the relevant frequencies for classification its better, cf bonus questions."*
> (번역) 필터링은 좋지만 데이터 맥락에서 의미가 있어야 함. 운동상상의 유의미한 주파수(~8–40Hz)가 유지되는지 확인. 분류에 적합한 주파수를 스스로 선택하면 더 좋음(보너스 참조).

**🗣️ 말하기:** "핵심 체크는 의미 있는 대역을 남겼나입니다. 통과 대역은 **7–30Hz**로 뮤(8–12)·베타(13–30) — ERD가 사는 대역 — 를 보존합니다. 아무 대역(예: 50–60Hz)을 남기면 신호는 있어도 운동상상엔 무의미하죠. 7–30Hz는 *데이터 맥락에서 의미*가 있습니다."
**🖥️ 시연:** `FMIN = 7.0, FMAX = 30.0`(`config.py`); 그리고 3.1의 필터 후 PSD.
> "64×321→4 *압축*(CSP) 자체는 **Implementation** 항목에서 채점되고, '주파수를 학습으로 선택'은 보너스 **FBCSP**로 보너스 항목에서 시연합니다."

**❓ Q&A**
- *왜 30까지만, 40 아니고?* — ERD가 8–30Hz에서 제일 강하고, 30Hz 위는 잡음 대비 신호가 낮아짐. 8–30Hz는 BCI 표준이고 0.658로 검증됨.
- *분류기에 실제로 들어가는 특징은?* — raw 필터 신호가 아니라 **CSP 로그-분산**(문제당 4숫자) = 판별 공간패턴에서의 ERD 세기.

→ **Yes**

## 3.3 Train
**📋 평가 기준 (평가표 원문 그대로):**
> *"The program has a train mode, sklearn score validation tools are used. The score for the training is displayed."*
> (번역) 프로그램에 train 모드가 있고, sklearn 점수 검증 도구를 쓰며, 학습 점수가 표시됨.

**🗣️ 말하기:** "`train` 모드가 `cross_val_score`를 돌려 점수를 출력하고 모델을 저장합니다."
**🖥️ 시연:** `python mybci.py 4 14 train`
```
[0.4444 1.0000 0.5556 0.6667 0.6667 1.0000 1.0000 0.6667 0.6667 0.6667]
cross_val_score: 0.7333
```
**👉 말하기:** "배열은 **교차검증 10 fold**의 정확도, 0.7333은 그 평균 = 학습 점수. 검증 도구는 `cross_val_score` + `ShuffleSplit(10, test_size=0.2)`: fold마다 36문제로 학습, 숨긴 9개로 시험이라 값이 k/9이고 평균으로 운을 평탄화. 이후 80%로 학습한 모델 + 남은 20%를 저장."

**❓ Q&A**
- *정확히 어떤 검증 도구?* — **전체 파이프라인**에 `cross_val_score`, `ShuffleSplit(n_splits=10, test_size=0.2)`.
- *누수 없나?* — 없음: CSP가 첫 단계라 fold마다 (CSP·LDA 둘 다) 학습분으로만 재학습.
- *`cross_val_score`가 한 번 학습하고 끝에 시험?* — 아님: 10라운드마다 **새 모델을 만들어** 채점하고 버림. 저장 모델은 80%에 별도 학습.
- *0.73인데 게이트는 60%?* — 60%는 **109명×6실험 평균(0.658)**에 적용, 한 사람이 아님.
- *재현?* — 기본은 분할이 매번 다름; `TPV_SEED=42`로 고정.

→ **Yes**

## 3.4 Predict
**📋 평가 기준 (평가표 원문 그대로):**
> *"There is a predict mode, which also uses validation tools. The prediction output is displayed (the id of the output class is enough)."*
> (번역) predict 모드가 있고 검증 도구도 사용하며, 예측 출력이 표시됨(클래스 id면 충분).

**🗣️ 말하기:** "`predict`가 저장된 모델을 불러와, 학습에 안 쓴 20%(9문제)를 하나씩 재생하며 예측과 정답을 비교합니다 — 재학습 없음."
**🖥️ 시연:** `python mybci.py 4 14 predict`
```
epoch 00:  [2]  [2] True
...
Accuracy: 0.6667
```
**👉 말하기:** "줄마다 **[예측] [정답] 일치**; id 1=상상 양손(T1), 2=상상 양발(T2). 이 9문제는 **학습 때 한 번도 못 본** 것이라 정직한 예측; 6/9 = 0.6667."

**❓ Q&A**
- *'검증 도구'는 어디?* — **held-out 검증셋**으로 돌리고 정답과 비교해 정확도 계산.
- *왜 9문제만?* — `train`이 이 피험자/실험에서 빼둔 20%.
- *왜 train 0.7333 vs predict 0.6667?* — 전자는 10-fold 평균, 후자는 고정 9문제 한 번. 표본 작고 추정 방식 다름.
- *predict가 재학습?* — 아니요, `.joblib` 로드만. 추론은 CSP 변환 + LDA 내적 한 번.
- *predict는 train과 같은 subject/run?* — 네: 산출물이 피험자/run별로 저장되고, 없으면 먼저 train하라는 에러. 모델은 사람별.

→ **Yes**

## 3.5 Realtime
**📋 평가 기준 (평가표 원문 그대로):**
> *"The prediction is made as the data is streamed to the processing pipeline. The program outputs the result between 0 and 2 seconds after the event was triggered."*
> (번역) 데이터가 파이프라인으로 스트리밍되는 동안 예측이 이루어지고, 이벤트 후 0~2초 사이에 결과를 출력.

**🗣️ 말하기:** "`predict`가 데이터를 **한 이벤트(2초 에폭)씩** 파이프라인에 흘려보냅니다(스트림처럼). 이벤트당 지연은 약 0.1–0.5 **밀리초**, 코드의 `assert latency < 2.0`이 예산을 강제."
**🖥️ 시연:** 이벤트당 지연 `0.0001–0.0005초  OK (<2s)`.
**🗣️ '2초'의 뜻:** "이벤트 후 2초 창(TMIN=0…TMAX=2.0)을 모아 에폭을 만들고, 추론은 1밀리초 미만이라 결과가 2초 안에 나옵니다."
**🗣️ 솔직하게:** "데이터가 사전 녹음이라 **시뮬레이션 스트림**입니다 — 라이브 하드웨어 대신 held-out 에폭을 하나씩 재생. `mne-realtime` 대신 루프로 구현했고, 외부 실시간 의존성 없이 '스트림, 2초 내'를 충족."

**❓ Q&A**
- *진짜 라이브 스트림?* — 아니요, 사전 녹음에 대한 이벤트 단위 시뮬레이션; 처리 모델(한 번에 하나, 2초 예산)은 라이브와 동일.
- *왜 그렇게 빠른가?* — 추론이 행렬곱 + 분산 + 내적뿐, 반복 계산 없음.
- *2초는 어디서 보장?* — `predict.py`의 `assert latency < LATENCY_BUDGET_S`, 그리고 에폭 자체가 이벤트 후 2초 창.

→ **Yes**

## 3.6 Integration
**📋 평가 기준 (평가표 원문 그대로):**
> *"Implementation was integrated to sklearn pipeline, inheriting from the baseEstimator and transformerMixin classes of sklearn."*
> (번역) 구현이 sklearn 파이프라인에 통합되고, sklearn의 BaseEstimator·TransformerMixin을 상속.

**🗣️ 말하기:** "`class MyCSP(TransformerMixin, BaseEstimator)`로 변환기 규약(`fit`/`transform`)을 갖춰 `Pipeline([('csp', MyCSP()), ('clf', LDA())])`에 그대로 들어갑니다."
**🖥️ 시연:** `isinstance(csp, BaseEstimator)=True`, `isinstance(csp, TransformerMixin)=True`, `get_params()={...}`, `fit_transform` 있음, `clone()` 동작.
**👉 말하기:** "BaseEstimator가 get/set_params(복제·튜닝), TransformerMixin이 fit_transform을 줍니다. 그래서 `cross_val_score`가 **fold마다 복제·재학습** 가능 — 이 상속이 누수 없는 채점의 토대지 형식이 아님."

**❓ Q&A**
- *왜 두 base class 다?* — BaseEstimator=파라미터 관리(복제/튜닝/검사), TransformerMixin=변환기 규약. 둘 다라야 온전한 sklearn 부품.
- *왜 `__init__`에서 계산 안 함?* — sklearn 규약: `__init__`은 하이퍼파라미터 저장만, 학습은 `fit`에서; 안 그러면 `clone`이 깨짐.
- *라이브러리 CSP 피했나?* — 네; `mne.decoding.CSP`는 테스트의 패리티 비교용뿐.

→ **Yes**

## 3.7 Implementation — 차원 축소 (배점 큰 항목)
**📋 평가 기준 (평가표 원문 그대로):**
> *"A dimensionality reduction algorithm is implemented, the subject talks about PCA and CSP but other algorithms performing a dimensionnality reduction are feasible. Check that the student has a general understanding of the algorithm. It is allowed to use functions from libs like numpy or scipy for some tasks : the eigenvalues decomposition, singular values decompositon and covariance matrix estimation."*
> (번역) 차원 축소 알고리즘이 구현됨(PCA/CSP 등). 학생이 알고리즘을 전반적으로 이해하는지 확인. 고유값분해·SVD·공분산 추정엔 numpy/scipy 사용 허용.

**🗣️ 말하기 + 🖥️ 시연:** "`csp.py`에 **CSP를 밑바닥부터** 구현. import는 `numpy`, `scipy.linalg.eigh`(허용), `sklearn.base`, 자작 `generalized_eigh`뿐 — **라이브러리 CSP 없음**."

**🗣️ 알고리즘 설명(2.7 요약):** (1) 답이 출렁임 크기에 있고 45문제로 2만차원 못 그으니 줄임; (2) 클래스별 공분산 C1, C2; (3) `eigh(C1, C1+C2)`로 λ=a/(a+b)가 0.5에서 가장 먼 필터들; (4) `@`로 섞어 로그-분산 → 4숫자.

**🖥️ 검증:** `pytest tests/test_csp_parity.py -v` → **PASSED** — 같은 분할(exp3)에서 `mne.decoding.CSP`와 0.05 이내 일치, 단일 피험자 ≥ 0.60.

**🗣️ 코드 투어(`csp.py`, 깊은 질문 대비):**
- `__init__`(13–17): 하이퍼파라미터만 저장.
- `_class_cov`(19–35): `E@E.T`(문제별 공분산) → **trace 정규화**(센 문제가 평균 지배 못하게) → 평균 → 대칭화(부동소수점) → **shrinkage**(`reg=0.01`, 대각으로 살짝 섞어 `eigh` 안정).
- `fit`(37–66): `y` 필수(지도) → C1,C2 → `eigh(C1,C1+C2)` → `|λ-0.5|` 내림차순 정렬 → `filters_`(상위 4×64); `patterns_`(유사역행렬, 시각화).
- `transform`(68–78): `filters_ @ E` → 분산 → 정규화 → log → 4숫자.

**❓ Q&A**
- *여기서 고유값은?* — 그 필터 출렁임 중 한 클래스의 비율; 0.5에서 멀수록 판별력↑.
- *여기서 공분산은?* — 클래스 안에서 채널이 같이 출렁이는 64×64 표; CSP 재료.
- *왜 일반화 고유문제(두 행렬)?* — 비율 `wᵀC1w/wᵀ(C1+C2)w`를 최대화하니 분모에 (C1+C2).
- *왜 CSP가 PCA보다?* — PCA=비지도(큰 변화, 잡음에 낚임), CSP=지도(클래스 차이).
- *왜 분산이 아니라 로그-분산?* — 종 모양으로 펴고 곱셈 차이를 직선 분류기용 덧셈 차이로.
- *`filters_` vs `patterns_`?* — 전자=신호→특징(분석), 후자=특징→두피(해석/topomap).
- *MNE CSP와 차이?* — 같은 알고리즘; 정확도 0.05 이내, 고유문제는 ~1e-14 일치(보너스 솔버).

→ **Yes**

## 3.8 Score (60% 게이트)
**📋 평가 기준 (평가표 원문 그대로):**
> *"There has to be a script executing training over each subject and computing the mean of scores over each subjects, by type of experiment runs. The mean of the resulting six means (corresponding to the six types of experiment runs) has to be superior or equal to 60%."*
> *(등급)* *"Over 60% add a point for every 1%."* — *"Rate it from 0 (failed) through 5 (excellent)"*
> (번역) 각 피험자에 학습을 돌리고 실험 유형별로 점수 평균을 구하는 스크립트가 있어야 함. 그 6개 평균의 평균이 60% 이상이어야 함. (등급) 60% 위로 1%마다 1점, 0~5점.

**🗣️ 말하기:** "`python mybci.py`(인자 없음)가 109명×6실험을 교차검증. 실험마다 피험자 평균(6개) → 그 6개의 평균."
**🖥️ 시연:** `python scripts/validate_60.py` (디펜스에선 **전체** 실행; `--fast 5`는 미리보기뿐).
```
experiment 0..5:  실험별 평균
Mean accuracy of 6 experiments:  0.658
```
**🗣️ 말하기:** "전체 109명 = **0.658 ≥ 0.60 → 통과**. 등급: 65.8%는 60% 위 5.8점 → 상한 **5/5**."

**❓ Q&A**
- *60% 계산?* — 실험별 평균(6개) → 그 6개의 평균. 한 번에 평탄 평균이 아님.
- *왜 `--fast 5`는 ~0.71인데 전체는 0.658?* — 앞 5명이 우연히 쉬움; 전체 109명엔 어려운 분도. 공식은 전체.
- *실패 피험자?* — `run_all`은 데이터/IO 오류만 좁게 skip; 진짜 알고리즘 버그는 터뜨림(숨기지 않음).
- *시간?* — 캐시 시 ~2.5분; 최초엔 ~3.1GB 다운로드도.

→ **Yes / 5/5**

## 3.9 Bonus · Datasets
**📋 평가 기준 (평가표 원문 그대로):**
> *"Are there other datasets processed by the program ? Is the scoring on those datasets correct ? Try to assert this taking into account the noise and the general quality of the dataset compared to the one given in the subject."*
> (번역) 다른 데이터셋도 처리하는가? 채점이 올바른가? 주제 데이터셋 대비 노이즈·품질을 고려해 판단.

**🗣️ 말하기:** "moabb로 **BCI Competition IV-2a**를 추가 — 구조가 다른 데이터셋(**22채널, 250Hz, 288 trial**)인데 파이프라인이 **그대로** 돌아갑니다."
**🖥️ 시연:** `python scripts/bonus_demo.py` → `[G] 2nd dataset BCI IV-2a (subj 1) : 0.8552`(모양 288×22×1001).
**🗣️ 말하기:** "그대로 도는 건 **MyCSP가 채널 수 무관**(공분산이 채널수²)이고, `external.py`가 moabb에 7–30Hz를 넘기기 때문. **0.855 > 0.658은 IV-2a가 정제된 대회 데이터(노이즈 적음)**라서 — 기준의 품질 고려 그대로. 프로젝트 `mne_data/`로 받고 필수 게이트와는 별도 실행."

**❓ Q&A**
- *64채널 코드가 왜 22채널에서?* — 채널 수 가정이 코드에 없음; 공분산·고유문제가 입력에 맞춰짐.
- *0.855 > 0.658이 의심?* — 아님: 깨끗한 데이터 → 높은 점수; 기준도 품질 고려를 요구.
- *전처리 같나?* — 네, moabb 패러다임으로 7–30Hz; 같은 왼손vs오른손 상상 과제.

→ **Yes**

## 3.10 Bonus · Feature engineering
**📋 평가 기준 (평가표 원문 그대로):**
> *"Try to evaluate the relevance of the preprocessing stage and how are the data feeded to the algorithm. The use of fourier or wavelet transform, and anything that transform the data before the processing is a plus."*
> (번역) 전처리 단계의 적절성과 데이터가 알고리즘에 어떻게 입력되는지 평가. 푸리에/웨이블릿 변환 등 처리 전 변환은 가산점.

**🗣️ 말하기:** "주파수 변환이 두 층. (1) 필수 **FIR 밴드패스**가 이미 주파수 영역 변환. (2) 보너스 **Filter-Bank CSP(FBCSP)**가 7–30Hz를 **4개 하위 밴드(8-12 / 12-16 / 16-20 / 20-30)**로 쪼개 밴드마다 CSP → 8특징."
**🗣️ 왜 쪼개나(깊이):** "단일 7–30Hz는 뮤·저베타·고베타를 한 덩어리로 뭉침. 하지만 **ERD 세기·분포는 주파수마다 다름** — 뮤 우세, 베타 우세인 사람이 다름. FBCSP는 **밴드마다 공간패턴을 따로 학습**하고, 분류기가 **어느 밴드가 중요한지 가중**. 단일 CSP에 없는 주파수 해상도. 푸리에/웨이블릿 같은 주파수 영역 특징공학 계열."
**🗣️ 데이터 입력 방식:** "각 하위 밴드 → 그 밴드 CSP → 로그-분산 → **이어붙여** 분류기에. 밴드패스가 `fit`/`transform` 안에 있어 `cross_val_score`가 fold마다 재학습(누수 없음)."
**🖥️ 시연:** `[F] Filter-Bank CSP (4 sub-bands) : 0.8222`(기본 CSP 0.8444).
**🗣️ 솔직하게:** "여기선 0.822 < 0.844 — 작은 데이터에 8특징이면 약간 과적합. FBCSP는 엄밀히 **더 표현력** 있고 다른 피험자/데이터셋에선 이길 수 있음; 더 정교한 특징공학 시연."

**❓ Q&A**
- *FBCSP가 단일 CSP보다 더 하는 일?* — 밴드별 공간패턴, 분류기가 밴드 중요도 학습.
- *왜 항상 더 정확하지 않나?* — 특징 증가 → ~45문제에서 약간 과적합; 보장된 승리가 아니라 정교함 시연.

→ **Yes**

## 3.11 Bonus · Implementations
**📋 평가 기준 (평가표 원문 그대로):**
> *"How deep did the student dig into his implementation ? ( Did he implement his own eigenvalues decomposition, SVD, or covariance matrix estimation ? ) ( Did he implement a complex dimensionality reduction algorithm ? ) Is there some kind of hyperparameter tuning or learning ? Did he implement his own classifier ?"*
> (번역) 구현을 얼마나 깊이 팠나? (자작 고유값분해/SVD/공분산 추정?) (복잡한 차원축소?) 하이퍼파라미터 튜닝/학습? 자작 분류기?

### A. 자작 고유분해(Jacobi) — 자작 고유값분해
**🗣️ 말하기:** "`eigh`마저 손으로 — `jacobi.py`의 **순수 numpy cyclic Jacobi** 솔버."
**🗣️ 어떻게 도나(깊이):** "비대각 원소를 한 쌍씩 회전으로 0으로, 대각이 될 때까지 sweep — 대각이 고유값, 누적 회전이 고유벡터. **일반화** `C1 w = λ(C1+C2)w`는 **백색화**로 표준화 후 Jacobi 두 번. `+,*,sqrt,sign,matmul`만 — `np.linalg.eig/eigh/svd`·scipy 안 씀."
**🖥️ 시연:**
```
[A] Jacobi 정확도            : 0.8444   (scipy eigh와 동일)
    표준 고유분해 vs numpy   : 5.33e-14   ← 기계 정밀도
    일반화 고유분해 vs scipy : 1.55e-15
```
`MyCSP(solver="jacobi")`로 켬.

### C. 하이퍼파라미터 튜닝 — 튜닝/학습
**🗣️ 말하기:** "`evaluate.tune`에서 `GridSearchCV`가 `n_components ∈ {4,6,8}`을 탐색, inner fold마다 CSP 재학습 — **누수 없는 중첩 CV**. 공식 게이트는 비교 공정성 위해 4 고정, 튜닝은 데모."
**🖥️ 시연:** `[C] tuned {'csp__n_components': 8} -> cv 0.8444`.

### D. 자작 분류기(OwnLDA) — 자작 분류기
**🗣️ 말하기:** "`own_lda.py`에 LDA도 손으로: 풀드 공분산, **`w = Σ⁻¹(μ1−μ0)`**, 사전확률 보정 절편, 판정 = sign(`w·x+b`), `np.linalg.solve`만."
**🖥️ 시연:**
```
[D] OwnLDA 정확도          : 0.7889
    sklearn LDA와 일치율    : 100.0%
```
`build_pipeline(clf="own-lda")`로 켬.

(+ **F. FBCSP**도 "복잡한 차원축소" 항목에 해당.)

**❓ Q&A**
- *정말 고유분해 라이브러리 없이?* — `jacobi.py`는 기본 연산만; scipy와 ~5e-14 일치.
- *일반화 문제는 어떻게?* — 백색화로 표준화 후 Jacobi 두 번.
- *튜닝 누수 없나?* — 없음; inner fold마다 CSP 재학습.
- *OwnLDA가 sklearn과 같나?* — 예측 100% 동일(같은 풀드 공분산 + Σ⁻¹(μ1−μ0)).

→ **Yes**

## 3.12 Ratings
**📋 평가표 원문 그대로:**
> *"Don't forget to check the flag corresponding to the defense"* — *"Ok"* / *"Outstanding project"*
> 플래그: *"Empty work / Incomplete work / Cheat / Crash / Forbidden function"*
> (번역) 디펜스에 해당하는 플래그 체크 잊지 말 것 — Ok / Outstanding project. 플래그: 빈 작업/미완성/부정행위/크래시/금지된 함수.

**🗣️ 말하기:** "부정 플래그(빈 작업/미완성/부정행위/크래시/**금지된 함수**) 하나도 해당 없음. 특히 기준이 고유값분해·SVD·공분산에 numpy/scipy를 명시 허용했는데, 저는 그것조차 직접 구현했습니다. 필수 0.658 통과 + 보너스 5개 전부 구현·검증 → **Outstanding project** 제안."
→ **Outstanding**

## 3.13 Conclusion
**📋 평가표 원문 그대로:** *"Leave a comment on this evaluation ( 2048 chars max )"*
(번역) 이 평가에 코멘트를 남기세요(최대 2048자).

평가관이 코멘트를 적는 칸. 마무리 한 문장:
> "PhysioNet 64채널을 7–30Hz로 걸러 ERD를 드러내고, **직접 구현한 CSP**로 4숫자로 압축, LDA로 분류, **누수 없는 교차검증**으로 109명×6실험 평균 **0.658**(≥60%)을 달성했고, **자작 고유분해 포함 보너스 5개**를 전부 구현했습니다."

---

# 4. 명령어 런북 (순서대로)

```bash
source .venv/bin/activate

python scripts/validate_60.py              # 전체 60% 게이트 → 0.658  (디펜스 전에도 실행)
pytest -m "not network"                    # 빠른 단위 테스트
pytest tests/test_csp_parity.py -v         # 자작 CSP vs mne, 0.05 이내
python scripts/visualize.py 4 14           # raw vs 7–30Hz + PSD
python mybci.py 4 14 train                 # cross_val_score + 모델 저장
python mybci.py 4 14 predict               # held-out 문제 스트림
python mybci.py                            # 6실험 평균 → 0.658
python scripts/bonus_demo.py               # 보너스 A,C,D,F,G 한 번에
```

---

# 5. 최종 체크리스트

| 평가표 항목 | 상태 | 근거(실측) |
|---|---|---|
| Preprocessing (plot) | ✅ | raw vs 7–30Hz + PSD; 60Hz/드리프트 제거 |
| Feature extraction | ✅ | 뮤·베타(7–30Hz) 유지 |
| Train | ✅ | `cross_val_score` 0.7333 표시 |
| Predict | ✅ | held-out 6/9 = 0.667 출력 |
| Realtime | ✅ | 이벤트당 ~0.0005초 < 2초 |
| Integration | ✅ | BaseEstimator + TransformerMixin; clone 동작 |
| Implementation (차원축소) | ✅ | 자작 CSP; 패리티 테스트 PASS(<0.05) |
| Score | ✅ **5/5** | 0.658 ≥ 0.60 |
| Bonus: Datasets | ✅ | BCI IV-2a 0.855 (22ch/250Hz) |
| Bonus: Feature engineering | ✅ | FBCSP 0.822 (4 하위밴드) |
| Bonus: Implementations | ✅ | Jacobi 5e-14 · 튜닝 · OwnLDA 100% |
| Ratings | ⭐ Outstanding | 부정 플래그 없음 |

---

# 6. 교차 난문 뱅크 (커브볼)

**방법·설계**
- *왜 exp0(왼손 vs 오른손)이 제일 약한가(~0.57)?* — 둘 다 손이고, 인접한 거의 대칭인 운동영역이라 반대쪽 ERD 차이가 작음; 양손vs양발(뚜렷, 정중앙 vs 측면)은 훨씬 쉬움(~0.71–0.84).
- *피험자마다 정확도 편차가 큰데 버그?* — 아님, BCI는 사람 편차가 유명함("BCI illiteracy"); 그래서 게이트가 109명 평균이고, 사람마다 10 fold 평균.
- *딥러닝이 이기지 않나?* — 데이터가 훨씬 많고 사람별 캘리브레이션이면 가능하지만, 고전 CSP+LDA가 해석 가능·빠름·데이터 효율적인 표준이고 기준을 통과; 주제도 차원 축소 알고리즘을 요구.
- *점수에 제일 큰 위험은?* — 비판별·노이즈 큰 피험자; shrinkage 정칙화 공분산과 10 fold 평균으로 완화.

**정확성·엄밀성**
- *데이터 누수 없음을 증명해봐.* — CSP가 첫 단계라 `cross_val_score`가 fold마다 파이프라인 전체를 복제·재학습; 누수 vs 정상 차이(1.0 vs 0.844)를 보여줄 수 있음. FBCSP 밴드패스도 같은 이유로 fit/transform 안.
- *CSP가 그럴듯한 게 아니라 맞다고 증명해봐.* — `mne.decoding.CSP`와 패리티(같은 분할 정확도 <0.05); 보너스 Jacobi가 scipy와 ~1e-14.
- *왜 공분산을 trace 정규화?* — 한 고진폭 문제가 클래스 평균을 지배하지 않게; 문제들을 공평하게.
- *`reg` shrinkage는 왜?* — 공분산을 잘 조건화해 채널이 적/공선일 때도 `eigh` 안정.
- *0.658 재현?* — `TPV_SEED=42`로 분할 고정; 미고정 시 0.65 부근에서 약간 변동하지만 게이트는 grand mean.

**라이브러리 / '금지된 함수'**
- *금지된 함수 썼나?* — 아니요. 기준이 고유값분해/SVD/공분산에 numpy/scipy를 허용; scipy `eigh`·numpy 공분산을 쓰고, 둘 다 밑바닥부터 재구현(Jacobi, OwnLDA).
- *왜 `mne`·`scikit-learn`을 쓰나?* — 주제가 요구: MNE는 EEG 입출력/필터/에폭, sklearn은 파이프라인/검증. *알고리즘*(CSP)은 직접.

**데이터**
- *데이터 어디 저장?* — 프로젝트 `./mne_data/`(PhysioNet ~3.1GB, BCI IV-2a ~83MB), gitignore라 폴더와 함께 이동하되 커밋 안 됨.
- *한 에폭 모양과 이유?* — (64, 321): 64채널 × (2초 × 160Hz + 1)샘플, +1은 양 끝 포함.
- *T1/T2 뜻?* — 데이터셋의 동작 표시; 뜻은 run 그룹에 따름(예: run 3/7/11 → 왼/오른 주먹; 5/9/13 → 양주먹/양발).

**라이브에서 뭔가 실패하면**
- *한 피험자가 에러.* — `run_all`은 데이터/IO 실패만 skip하고 보고; grand mean은 나머지로. 그 피험자 `train`을 단독 재실행해 보일 수 있음.
- *보너스 데이터셋 인터넷 없음.* — `./mne_data/`에 미리 받아둠; 없으면 첫 `bonus_demo` 때 83MB 다운로드.

> **준비 끝.** 2부를 체화하고, 3부를 순서대로 걷고, 6부를 주머니에 넣어두세요.

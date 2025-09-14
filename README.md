# SarcopeniaFeatureSelection
A study on feature selection methods for a sarcopenia prediction model

## 1. Project Name & Acknowledgments
- **Project Name(KOR)**: 포노 사피엔스 시대의 시니어를 위한 건강관리 [Re:] 솔루션: 근감소 예방과 간리를 위한 헬스웨이 구축
- **Project Name(ENG)**: Health care [RE:] Solutions for Senior in the Phonosapiens Era : Building a Healthway for preventing and mansging Sarcopenia
- Multi-Agent Reinforcement Learning–based (MARL) feature selection
- This project is a collaborative initiative of the **Institute of Human Convergence Health Science (IHCHS)**, involving a partnership between the departments of **Social Science**, **Health Science**, and **Computer Science**.
- This project is conducted as part of a research initiative supported by the IHCHS and the [National Research Foundation of Korea](https://www.ntis.go.kr/ThSearchTotalList.do?sort=&ntisYn=&searchWord=2022S1A5C2A07090938&originalSearchWord=&originalSearchGubun=) (NRF, Project No. 2022S1A5C2A07090938).
---

## 2. Background
- Sarcopenia is a disease characterized by the loss of muscle mass, strength, and physical function. It is primarily driven by the aging process, often accompanied by contributing factors such as chronic diseases, malnutrition, and decreased physical activity.
- Sarcopenia was officially recognized as a disease with the assignment of a diagnostic code in the WHO's ICD-10-CM in 2016, and subsequently in South Korea's 8th Korean Standard Classification of Diseases (KCD-8) revision in 2021.
- In South Korea, its prevalence is estimated to be between 10-28% among the elderly population aged 65 and over.
- However, since it has only recently been classified as a disease, there is limited data for analysis.

---

## 3. System
<p align="center">
    <img width="600" height="253" alt="arch" src="https://github.com/user-attachments/assets/57d9626b-2020-4120-9ab0-27291d786ba7" />
</p>

- This dataset was collected from Korean older adults aged 65 to 90, and feature selection is performed among 44 (a subset of the total) features using MARL.
- We employed Double DQN (DDQN)
- State: Regarding the features selected by the agents. We plan to enhance state information by adding statistical measures (mean, std, median, IQR) to the agent-selected features in future datasets.
- Action: Each feature can either be selected or not selected. To reduce the action space of the agents, we employed a multi-agent approach.
- Reward: Each agent is given a different reward based on its contribution.
```math
Reward_i=(Acc_t-Acc_{t-1})\times\frac{|credit_i|}{\sum_j|credit_j|}
```

---

## 4. Runs

    ```
    pip install -r requirements.txt
    python3 main.py --episodes 5000 --batch_size 64 --dataset "breast_cancer" --cat_iterations 100
    ```

---

## 5. Preliminary Results
All experimental results are reported as k-fold accuracies.


### Binary classification
[Normal, Sarcopenia]

| | Without feature selection | Feature selection |
|---|---|---|
| Accuracy | 90% | 93% |
| F1 score | 84% | 88% |
| The num of features | 44 | 25 |

Confusion Matrix
<table>
  <tr>
    <td align="center"><b>Without FS</b></td>
    <td align="center"><b>FS</b></td>
  </tr>
  <tr>
    <td align="center"><img src="https://github.com/user-attachments/assets/4029bf58-094c-4854-91e9-e5f22c3738f2" alt="Without FS" width="600"></td>
    <td align="center"><img src="https://github.com/user-attachments/assets/7d97a42e-0c4d-4600-9636-ae7d9806e67c" alt="FS" width="600"></td>
  </tr>
</table>
 
 
### Multi class classification
[Normal, Possible, Sarcopenia, Severe]

| | Without feature selection | Feature selection |
|---|---|---|
| Accuracy | 76% | 84% |
| F1 score | 71% | 79% |
| The num of features | 44 | 18 |

Confusion Matrix
<table>
  <tr>
    <td align="center"><b>Without FS</b></td>
    <td align="center"><b>FS</b></td>
  </tr>
  <tr>
    <td align="center"><img src="https://github.com/user-attachments/assets/1b443f56-81dc-4e73-862f-896698880592" alt="Without FS" width="600"></td>
    <td align="center"><img src="https://github.com/user-attachments/assets/59fc2457-7c90-425e-a09d-f0d5e7ce09ae" alt="FS" width="600"></td>
  </tr>
</table>

---

## 6. License and Dataset
- Unfortunately, the dataset cannot be made publicly available.

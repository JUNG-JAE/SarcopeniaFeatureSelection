# SarcopeniaFeatureSelection
A study on feature selection methods for a sarcopenia prediction model

## 1. Project Name & Acknowledgments
<p align="center">
    <img width="300" height="160" alt="IHCHS" src="https://github.com/user-attachments/assets/49699cfe-fbb1-499b-b758-2c51c55ef81f" />
</p>

- **Project Name(KOR)**: 포노 사피엔스 시대의 시니어를 위한 건강관리 [Re:] 솔루션: 근감소 예방과 간리를 위한 헬스웨이 구축
- **Project Name(ENG)**: Health care [RE:] Solutions for Senior in the Phonosapiens Era : Building a Healthway for preventing and mansging Sarcopenia
- Multi-agent reinforcement learning–based feature selection
- This project is a collaborative initiative of the **Institute of Human Convergence Health Science (IHCHS)**, involving a partnership between the departments of **Social Science**, **Health Science**, and **Computer Science**.
- Supported by the [National Research Foundation of Korea](https://www.ntis.go.kr/ThSearchProjectList.do?searchCategory=project&encodingSearchWord=%25ED%258F%25AC%25EB%2585%25B8%2B%25EC%2582%25AC%25ED%2594%25BC%25EC%2597%2594%25EC%258A%25A4%25EC%258B%259C%25EB%258C%2580%25EC%259D%2598%2B%25EC%258B%259C%25EB%258B%2588%25EC%2596%25B4%25EB%25A5%25BC%2B%25EC%259C%2584%25ED%2595%259C%2B%25EA%25B1%25B4%25EA%25B0%2595%25EA%25B4%2580%25EB%25A6%25AC%2B%255BRe%253A%255D%2B%25EC%2586%2594%25EB%25A3%25A8%25EC%2585%2598&oldSearchWord=포노+사피엔스시대의+시니어를+위한+건강관리+[Re%3A]+솔루션&encodingOldSearchWord=%25ED%258F%25AC%25EB%2585%25B8%2B%25EC%2582%25AC%25ED%2594%25BC%25EC%2597%2594%25EC%258A%25A4%25EC%258B%259C%25EB%258C%2580%25EC%259D%2598%2B%25EC%258B%259C%25EB%258B%2588%25EC%2596%25B4%25EB%25A5%25BC%2B%25EC%259C%2584%25ED%2595%259C%2B%25EA%25B1%25B4%25EA%25B0%2595%25EA%25B4%2580%25EB%25A6%25AC%2B%255BRe%253A%255D%2B%25EC%2586%2594%25EB%25A3%25A8%25EC%2585%2598&resultSearchValue=&fileSearchYn=&sort=RANK%2FDESC&ntisYn=&dbt=project&init=&cordisKakenYn=&is01=&originalSearchWord=포노+사피엔스시대의+시니어를+위한+건강관리+[Re%3A]+솔루션&originalSearchGubun=&technologyClassification=&directorySearchYear=&directorySearchOption1=&directorySearchOption2=&directorySearchOption3=&searchWord=2022S1A5C2A07090938) (NRF).
- **Project Identification Number:** 2022S1A5C2A07090938

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

- This dataset was collected from Korean older adults aged 65 to 90, and feature selection is performed among 44 (a subset of the total) features using multi-agent reinforcement learning.
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
    python3 main.py -- ~~~~
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
    <td align="center"><img src="https://github.com/user-attachments/assets/59fc2457-7c90-425e-a09d-f0d5e7ce09ae" alt="Without FS" width="600"></td>
    <td align="center"><img src="https://github.com/user-attachments/assets/1b443f56-81dc-4e73-862f-896698880592" alt="FS" width="600"></td>
  </tr>
</table>

---

## 6. License and Dataset
- Unfortunately, the dataset cannot be made publicly available.

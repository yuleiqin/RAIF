---
license: mit
language:
- en
pretty_name: InfoBench
size_categories:
- n<1K
---


# Dataset Card for InFoBench Dataset

## Table of Contents
- [Dataset Description](#dataset-description)
- [Dataset Usage](#dataset-usage)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
- [Additional Information](#additional-information)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
 
## Dataset Description

- **Repository:** [InFoBench Repository](https://github.com/qinyiwei/InfoBench)
- **Paper:** [InFoBench: Evaluating Instruction Following Ability in Large Language Models](https://arxiv.org/pdf/2401.03601.pdf)

The InFoBench Dataset is an evaluation benchmark dataset containing 500 instructions and corresponding 2250 decomposed requirements.

## Dataset Usage
You can directly download it with huggingface datasets.
``` python
from datasets import load_dataset

dataset = load_dataset("kqsong/InFoBench")
```

## Dataset Structure
### Data Instances
For each instance, there is an instruction string, an input string (optional), a list of decomposed questions, and a list of the labels for each decomposed question.

```json
{
    "id": "domain_oriented_task_215",
    "input": "",
    "category": "Business and Economics: Business Administration",
    "instruction": "Generate a non-disclosure agreement of two pages (each page is limited to 250 words) for a software development project involving Party A and Party B. The confidentiality duration should be 5 years. \n\nThe first page should include definitions for key terms such as 'confidential information', 'disclosure', and 'recipient'. \n\nOn the second page, provide clauses detailing the protocol for the return or destruction of confidential information, exceptions to maintaining confidentiality, and the repercussions following a breach of the agreement. \n\nPlease indicate the separation between the first and second pages with a full line of dashed lines ('-----'). Also, make sure that each page is clearly labeled with its respective page number.",
    "decomposed_questions": [
        "Is the generated text a non-disclosure agreement?",
        "Does the generated text consist of two pages?",
        "Is each page of the generated text limited to 250 words?",
        "Is the generated non-disclosure agreement for a software development project involving Party A and Party B?",
        "Does the generated non-disclosure agreement specify a confidentiality duration of 5 years?",
        "Does the first page of the generated non-disclosure agreement include definitions for key terms such as 'confidential information', 'disclosure', and 'recipient'?",
        "Does the second page of the generated non-disclosure agreement provide clauses detailing the protocol for the return or destruction of confidential information?",
        "Does the second page of the generated non-disclosure agreement provide exceptions to maintaining confidentiality?",
        "Does the second page of the generated non-disclosure agreement provide the repercussions following a breach of the agreement?",
        "Does the generated text indicate the separation between the first and second pages with a full line of dashed lines ('-----')?",
        "Does the generated text ensure that each page is clearly labeled with its respective page number?"
    ],
    "subset": "Hard_set",
    "question_label": [
        ["Format"],
        ["Format", "Number"],
        ["Number"],
        ["Content"],
        ["Content"],
        ["Format", "Content"],
        ["Content"],
        ["Content"],
        ["Content"],
        ["Format"],
        ["Format"]
    ]
}
```


### Data Fields
- `id`: a string.
- `subset`: `Hard_Set` or `Easy_Set`.
- `category`: a string containing categorical information.
- `instruction`: a string containing instructions.
- `input`: a string, containing the context information, could be an empty string.
- `decomposed_questions`: a list of strings, each corresponding to a decomposed requirement.
- `question_label`: a list of list of strings, each list of strings containing a series of labels for the corresponding decomposed questions.

## Additional Information

### Licensing Information
The InFoBench Dataset version 1.0.0 is released under the [MIT LISENCE](https://github.com/qinyiwei/InfoBench/blob/main/LICENSE)

### Citation Information
```
@article{qin2024infobench,
      title={InFoBench: Evaluating Instruction Following Ability in Large Language Models}, 
      author={Yiwei Qin and Kaiqiang Song and Yebowen Hu and Wenlin Yao and Sangwoo Cho and Xiaoyang Wang and Xuansheng Wu and Fei Liu and Pengfei Liu and Dong Yu},
      year={2024},
      eprint={2401.03601},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
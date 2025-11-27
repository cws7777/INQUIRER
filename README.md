# INQUIRER: Harnessing Internal Knowledge GRaphs for Video Question Generation

<img width="1333" height="533" alt="image" src="https://github.com/user-attachments/assets/a00af011-17e5-4ad2-979f-1195535dd394" />

[Paper](https://www.sciencedirect.com/science/article/pii/S0950705125010780) from [Knowledge-Based Systems Journal](https://www.sciencedirect.com/journal/knowledge-based-systems) (IF. 7.6, Q1)

Authors: [Woo Suk Choi](https://cws7777.github.io/), [Youwon Jang](https://greeksharifa.github.io/about/), [Minsu Lee](https://scholar.google.com/citations?user=75_DkUwAAAAJ), [Byoung-Tak Zhang](https://scholar.google.com/citations?user=sYTUOu8AAAAJ&hl=en)

**Abstract**: Video question generation (VideoQG) aims to generate questions about video content to facilitate and assess video understanding. Existing works which primarily condition question generation on answer-related information such as the answer itself or its attributes. However, these methods are primarily designed as data augmentation techniques and thus struggle to produce semantically diverse questions. We propose INQUIRER, a novel VideoQG framework that leverages internal knowledge graphs derived from video information to generate meaningful and diverse questions. INQUIRER consists of three key steps: KCon, which constructs an internal knowledge graph to represent a video similarly to human knowledge structures, QGen which generates questions based on the video and the knowledge graph, and QCur which refines the generated questions to ensure quality and contextual relevance. Each generated question is accompanied by a correct answer and plausible distractors to support downstream QA evaluation. To comprehensively evaluate the generated QAs and utility of INQUIRER from multiple perspectives, we utilize widely used video question answering (VideoQA) benchmarks, including DramaQA, TVQA, How2QA, and STAR. Experiment results demonstrate that INQUIRER not only generates high-quality question–answer pairs but also significantly enhances VideoQA performance, validating its effectiveness as a robust framework for video question generation.

To clone the repository
```
git clone https://github.com/cws7777/INQUIRER.git
```

For downloading datasets for knowledge graphs of DramaQA, TVQA, How2,
please request to wschoi@bi.snu.ac.kr

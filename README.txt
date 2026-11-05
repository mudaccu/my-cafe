TMUC Unit 25: Machine Learning - From Model to App
Instructor: Hadeed Bin Toufique
Date: 03 Nov 2026

IMPORTANT: The campus cafeteria data in this lesson is SYNTHETIC and was invented for teaching.
Do not treat the output as a real operational estimate, nor claim its MAE represents a real cafe.

1. Download the lecture code folder (or extract the ZIP).
2. Open a terminal inside that folder (where app.py and train_model.py are stored).
3. On your own laptop with Python installed, run:
     python -m pip install -r requirements.txt
     python train_model.py
     python -m streamlit run app.py
4. Your browser opens the local app (commonly http://localhost:8501).
5. Change the numbers, press Predict waiting time and inspect the input table.
6. Stop the app with Ctrl+C.

Note: This is local Python/Streamlit, not a Google Colab shareable URL.
If you only have Colab, you can still read and run train_model.py's Python code there,
but the local Streamlit app requires a local Python runtime or a separately configured hosting environment.

CLASS ACTIVITIES
A. Write the feature names and target on the board before opening the code.
B. Identify train/test rows, and explain why the fitted pipeline is saved.
C. Run the app using two example inputs; compare outcomes. Do not promise monotonicity
   for every possible input because a fitted decision tree may behave non-monotonically.
D. As a team of 5, assign 1 leader, 1 operator, 1 tester, 1 evidence collector, 1 presenter.
E. Present what could go wrong in a real cafe: unusual demand, inconsistent measurement,
   unavailable features, historical drift, or input mistakes.

EVIDENCE: a screenshot of the dataset, training output, the opened app, and a written test case.
For individual TMUC coursework, students must use their own approved dataset and scope.

SAFETY: joblib uses pickle-like loading. NEVER load a downloaded/untrusted .joblib file.
Train the model yourself using the bundled code, or obtain it from a verified trusted source.
Model files can be incompatible across different scikit-learn versions.

Official references:
https://docs.streamlit.io/get-started/installation/command-line
https://docs.streamlit.io/develop/concepts/architecture/run-your-app
https://scikit-learn.org/stable/model_persistence.html

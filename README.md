# Generalized Linear Models from First Principles

This project provides the full mathematical derivation of Generalized Linear Models and an exemplification of them by the from-scratch implementation of logistic regression using only Python, NumPy, and SciPy. It focuses on deriving GLMs from first principles, emphasizing their probabilistic foundations (exponential family, link functions), the Iteratively Reweighted Least Squares (IRLS) estimation algorithm, and comprehensive statistical inference.

The custom implementation is successfully validated against R's glm() function, and applied to a Heart Disease Dataset( with interpretation of the results).

---

📘 **[Read the Full Theoretical Report (PDF)](GLMsFromScratch.pdf)**
Includes all required background, mathematical derivation of GLMs, derivations of the IRLS algorithm, and the mathematical foundations of the implementation.

---

The implementation provides a complete logistic regression estimator capable of performing binary classification. It includes essential inference tools like standard errors, confidence intervals, and automated stepwise variable selection, mirroring the functionality of industry-standard statistical software like R.

## Features

*   **Detailed Report:** A comprehensive theoretical derivation of GLMs, the IRLS algorithm, and implementation details.
*   **Logistic Regression from Scratch:** Core components built using only NumPy and SciPy for numerical operations.
*   **Iteratively Reweighted Least Squares (IRLS):** The model is fitted using a from-scratch implementation of the IRLS algorithm, identical to the Fisher Scoring method for canonical links.
*   **Full Statistical Inference:**
    *   Standard Errors, Wald Z-tests, and P-values.
    *   Confidence Intervals for coefficients and Odds Ratios.
    *   Model fit statistics: Log-Likelihood, Deviance, AIC, BIC, and McFadden's Pseudo R².
*   **Stepwise Variable Selection:** An automated backward elimination feature (`step()` method) using AIC, which correctly handles categorical predictors by grouping dummy variables.
*   **Validation:** The implementation is thoroughly validated against R's `glm()` function, showing identical results for coefficients, AIC scores, and feature selection steps.
*   **Data Preprocessing Pipeline:** A detailed Jupyter Notebook (`Preprocessing.ipynb`) documents all data cleaning, imputation, and transformation steps.


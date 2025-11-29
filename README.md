# Customer Churn Prediction & Retention Optimization

## Project Overview
Built an end-to-end machine learning solution to predict customer churn in subscription businesses, addressing the fundamental challenge of maintaining sustainable CLV/CAC ratios. Developed a Random Forest classification model and interactive Streamlit dashboard that transforms customer data into actionable retention strategies.

### Key Technical Work:

- Engineered predictive model using Python, scikit-learn, and pandas to identify at-risk customers before they churn
Conducted threshold optimization analysis to balance churn detection (81-87%) against operational contact capacity (38-51% of customer base)
- Built interactive visualizations demonstrating business trade-offs between detection accuracy and resource constraints
- Designed framework for cost-benefit analysis incorporating campaign costs, CLV, and operational capacity

### Business Impact:
The threshold analysis framework enables companies to make data-driven decisions about retention campaign scope. For example, at a 0.4 threshold, the model flags 42% of customers while catching 81% of potential churners, allowing businesses to allocate retention resources strategically rather than reactively. This approach recognizes that preventing churn directly extends customer lifetime value and improves acquisition efficiency by reducing the need to constantly replace departing customers.

### Tools: Python, scikit-learn, pandas, matplotlib, Streamlit

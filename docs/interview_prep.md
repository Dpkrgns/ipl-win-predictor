# Interview Preparation

## 20 likely interview questions with strong answers

1. What problem are you solving?
   I predict the batting team's probability of winning during the second innings of an IPL match. Each row represents a live chase state after a ball, and the target is whether the chasing team eventually won.

2. Why only the second innings?
   A chase has a known target, so features like runs left, balls left, and required run rate become directly meaningful. First innings prediction is a different problem because the final par score is uncertain.

3. What is the target variable?
   `batting_team_won` is 1 when the second-innings batting team equals the match winner, otherwise 0. It is created after merging second-innings ball states with the match result.

4. Why use ball-by-ball data instead of match-level data?
   Ball-by-ball data lets the model learn changing game states. Match-level data only gives one row per match and cannot support live probability prediction.

5. What are the most important engineered features?
   Runs left, balls left, wickets remaining, required run rate, current run rate, run rate gap, and pressure index. They capture resources, demand, and recent match pressure.

6. Why is wickets remaining useful?
   More wickets give the batting team freedom to take risks. Low wickets reduce scoring flexibility even when runs left are manageable.

7. Why can required run rate be misleading alone?
   A required rate of 10 is very different with 8 wickets in hand versus 2 wickets in hand. That is why interaction features such as pressure index help.

8. What is pressure index?
   It combines required run rate, wickets lost, and overs remaining. It rises when the chase needs fast scoring with fewer resources.

9. How did you avoid data leakage?
   The features only use information available at that ball. The target uses final outcome, but final outcome is not included as a feature. Splitting by `match_id` avoids balls from the same match appearing in both train and test.

10. Why split by match instead of random rows?
   Random row splitting leaks match context because one match contributes many similar rows. Group splitting tests generalization to unseen matches.

11. Why compare Logistic Regression?
   It is a strong baseline, interpretable, and good for explaining linear relationships such as higher required run rate lowering win probability.

12. Why use Random Forest?
   It captures nonlinear relationships and interactions, such as required run rate becoming more damaging when wickets are low.

13. Why use XGBoost or LightGBM?
   Gradient boosting often performs well on structured tabular data because it builds many small trees that correct earlier mistakes.

14. Which metric matters most?
   ROC-AUC is useful for probability ranking across thresholds, while F1 balances precision and recall. For a probability product, calibration would also be valuable.

15. What does precision mean here?
   Among states predicted as likely wins, precision measures how often the chasing team actually wins.

16. What does recall mean here?
   Among all actual successful chases, recall measures how often the model identifies them as winning states.

17. What does ROC-AUC mean in this project?
   It measures whether the model ranks true winning chase states above losing chase states across thresholds.

18. How do you explain a prediction?
   Use global feature importance to explain overall drivers, then SHAP values to show which features increased or decreased a specific prediction.

19. What are limitations of this project?
   It does not include batter quality, bowler quality, dew, toss impact, pitch condition, or player injuries. Venue is included, but richer context would improve realism.

20. How would you improve it?
   Add player-level form, phase-specific features, calibration curves, time-based validation, richer weather/venue signals, and a model monitoring dashboard.

## Common mistakes candidates make

- Treating every ball as independent while using random row split. This inflates metrics because match states leak across train and test.
- Saying accuracy is enough. Win probability models need ranking quality, probability quality, and threshold-aware metrics.
- Forgetting cricket intuition. Features should be explained in terms of resources, pressure, risk, and chase demand.
- Using final score or future wickets as features. That is direct leakage.
- Not explaining why second innings is a cleaner prediction task.
- Claiming the app is the ML project. The app is only the interface; the ML value is in data design, validation, evaluation, and interpretation.
- Ignoring team-name changes across seasons.
- Not handling wides and no-balls when computing balls left.
- Training one model without a baseline.
- Presenting feature importance without explaining direction or limitations.

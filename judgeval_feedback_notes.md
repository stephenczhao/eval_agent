# User Feedback and Potential Improvements

## [Onboarding](https://app.judgmentlabs.ai/app/onboarding):
![Environment Variable Screenshot](./images/feedback_imgs/env_variable_screenshot.png)

- The instructions for setting the environment variable could be clearer. While it's implied that the `***` is replaced with the real API key, it might look like a placeholder, leaving users unsure about where to obtain their actual API keys.
  - Possible Solution: Add a caption clarifying that the real API keys are displayed here and will be copied to the clipboard when the copy button is clicked.


## [Online Evaluations Documentations Missing](https://docs.judgmentlabs.ai/quickstarts#online-evaluation)
![screenshot1](./images/feedback_imgs/online_evaluations.png)

<div style="text-align: center; font-size: 3em;">
⬇️
</div>

![screenshot2](./images/feedback_imgs/oneline_evaluations_404.png)


# SDK Improvements: 

## JudgmentClient.run_evaluation Default Behavior

Currently, if you run the demo evaluation (`./demo_test.py`) twice, you will get the following error using default args: 

```
Traceback (most recent call last):
  File "/Users/stephenzhao/In Progress/Judgement_Labs/eval_agent/demo_test.py", line 18, in <module>
    results = client.run_evaluation(
              ^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/stephenzhao/In Progress/Judgement_Labs/judgeval/src/judgeval/judgment_client.py", line 213, in run_evaluation
    raise ValueError(
ValueError: Please check your EvaluationRun object, one or more fields are invalid: 
Eval run name 'default_eval_run' already exists for this project. Please choose a different name, set the `override` flag to true, or set the `append` flag to true.
```






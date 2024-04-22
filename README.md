# ASFI_formalization

NLP pipeline for ["Do We Run How We Say We Run? Formalization and Practice of Governance in OSS Communities"](https://arxiv.org/pdf/2309.14245.pdf) accepted in Proceedings of the CHI Conference on Human Factors in Computing Systems [CHI '24], May 11-16, 2024, Honolulu, HI, USA.

##Data

Data [Emails, Policies, Analysis] can be downloaded here 

## Organization

Scripts: 
```
	**email_filtering**: filtering emails for reporting reminders and JIRA/Github notifications
	**srl_parsing**: parsing activities/rules from email [column: 'reply'] and policies [column: 'policy.statement']
	**dev_speak**: remove logs/traces and other non-english dialogue
	**internalization_scoring**: calculate semantic similarity between rules and routine activities
```

Notebooks: 
```
	**cluster_tuning**: Select best clustering model
	**topic_prediction**: Predict topics for all governed activities
```
## Reproduction 

Experiments were conducted on older versions of AllenNLP/AllenNLP-models no longer compatible with Colab. Repo has been updated to 2.10.1. Minor differences in parsing may be observed due to version differences.

AllenNLP is now in maintenance mode only. Check here for forks: <br/>
* [AllenNLP](https://github.com/Mahasweta-usc/allennlp) <br/>
* [ALlenNLP-models](https://github.com/Mahasweta-usc/allennlp-models)
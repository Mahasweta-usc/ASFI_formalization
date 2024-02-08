# ASFI_formalization

NLP pipeline for "Do We Run How We Say We Run? Formalization and Practice of Governance in OSS Communities" accepted in Proceedings of the CHI Conference on Human Factors in Computing Systems (CHI '24), May 11-16, 2024, Honolulu, HI, USA.

##Data

Data (Emails, Policies) can be downloaded here 

###Organization

Scripts: 
	*email_filtering*: filtering emails for reporting reminders and JIRA/Github notifications
	*srl_parsing*: parsing activities/rules from email (column: 'reply') and policies (column: 'policy.statement')
	*dev_speak*: remove logs/traces and other non-english dialogue
	*internalization_scoring*: calculate semantic similarity between rules and routine activities

	Note: Original experiments were conducted on AllenNLP v.2.1.0, which is no longer compatible with the SRL pipeline. Repo has been updated to 2.10.1. 

Notebooks: 
	*cluster_tuning*: Select best clustering model
	*topic_prediction*: Predict topics for all governed activities
 
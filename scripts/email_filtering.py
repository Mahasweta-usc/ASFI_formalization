import pandas as pd

recurring_clauses = ['the incubator pmc requires your report to be submitted 2 weeks before the board meeting, to allow sufficient time for review and submission',
  'submitting your report',
 'again the very latest you should submit your report is 2 weeks prior to the board meeting',
 'this email sent by an automated system on behalf of the apache incubator pmc',
 'the report for your project will form a part of the incubator pmc report',
 'please submit your report with sufficient time to allow the incubator pmc, and subsequently board members to review and digest',
 'the incubator pmc requires your report to be submitted one week before the board meeting, to allow sufficient time for review',
 'this is an automatically generated e-mail']

recurring = ['@git.apache.org','jira@apache.org','noreply@github.com','notifications@github.com']



all_data = pd.read_csv('all_activities.csv')
org_columns = ['message_id','body','month']

projects = all_data['project_name'].unique(); print(len(projects))
all_data= all_data[all_data['list_name'].apply(lambda x : 'dev' in x)]
print(all_data['list_name'].unique())


all_data.dropna(subset=org_columns,inplace=True);print(all_data.shape[0])
all_data.drop_duplicates(subset=org_columns,inplace=True);print(all_data.shape[0])
all_data = all_data[all_data['from_commit']==False];print(all_data.shape[0])
all_data = all_data[all_data['is_bot']==False];print(all_data.shape[0]);print("Projects:", len(all_data['project_name'].unique()))

print(all_data.columns, all_data.shape[0])

# #filtering for reporting/notification emails
# all_data['is_report'] = all_data['body'].apply(lambda x : any(clause in x.lower() for clause in recurring_clauses))
# all_data = all_data[all_data['is_report']==False];print(all_data.shape[0])

# all_data = all_data[org_columns];all_data.reset_index(inplace=True)

# recurring = ['@git.apache.org','jira@apache.org','noreply@github.com','notifications@github.com']
# all_data['gitmail'] = all_data['sender_email'].apply(lambda x : True if any(elem in x for elem in recurring) else False)
# all_data = all_data[~all_data['gitmail']]
# print(all_data.columns, all_data.shape[0])



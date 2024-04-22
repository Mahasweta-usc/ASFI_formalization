#R2 (Tjur) and AIC is not supported by statsmodel and was calculated/reported separately using R

#topics maps for primary analysis
topic_assigned = {'0_release_releases_version' : 'Release Management',
                  '2_issue_fix_problem' : 'Issues',
                  '3_apache_incubator_dubbo' : 'Apache Incubator',
                  '4_security_user_password' : 'Security/Authentication',
                  '5_email_mail_mailing' : 'Email Communications',
                  '7_commit_committers_committer' : 'Committers/Commits',
                  '8_patch_patches_attached' : 'Patches',
                   '10_date_day_week' : 'Schedules/Events',
                   '12_help_got_doing' : 'Collaboration/Help',
                   '13_jira_browse_jiras' : 'JIRA',
                   '14_license_notice_copyright' : 'Copyright Notice',
                   '15_files_file_directory' : 'Project Directory',
                   '16_build_builds_built' : 'Builds',
                   '18_asf_icla_projects' : 'Apache Foundation/Contributor License',
                   '19_svn_subversion_diff' : 'Subversion (SVN)',
                   '20_vote_votes_voting' : 'General Voting',
                   '22_configuration_config_configure' : 'Project Configuration',
                   '30_report_reports_month' : 'Incubator Reporting',
                   '33_meeting_meetup_conference' : 'Apache Meetups/Conferences',
                   '35_task_tasks_job' : 'Tasks Handling',
                   '45_pmc_ppmc_members' : 'Podling Project Management Committee (PPMC)',
                   '57_wiki_page_mediawiki' : 'Project Wiki',
                   '61_install_installation_ubuntu' : 'Software Installations',
                   '64_delete_remove_removed' : 'Deletion/Removal',
                   '66_shutdown_process_run' : 'Processes',
                   '67_request_response_reply' : 'General Communications',
                   '69_check_checked_checks' : 'Checks/Tests',
                   '81_artifacts_artifact_nexus' : 'Artifact Management',
                   '87_list_lists_whimsy' : 'Mailing Lists/Whimsy',
                   '89_graduation_tlp_address' : 'Graduation Requirements/Maturity Model',
                   '91_community_open_source' : 'Community',
                   '98_looks_good_zoom' : 'Visibility/Resolution',
                   '102_project_projects_developers' : 'Project/Developers',
                   '107_wait_blocker_blockers' : 'Release Blockers',
                   '114_proposal_solution_task' : 'Proposals/Resolutions',
                   '121_doc_document_docs' : 'Documentation',
                   '123_resource_resources_resourcetype' : 'Resource Object Management',
                   '150_hours_vote_open' : 'Voting Protocol/Timeline',
                   '163_form_onsubmit_forms' : 'Forms',
                   '167_ipmc_votes_vote' : 'Incubator Project Management Committee (IPMC)',
                   '182_link_links_linked' : 'Links/URLs',
                   '188_incubator_incubation_incubating' : 'Incubation Process' }

# #topics for secondary or posthoc analysis

# topic_assigned = {'0_think_thank_good' : 'Community',
#                    '1_release_version_releases' : 'Release Management',
#                    '2_issue_fix_problem' : 'Issues',
#                    '4_apache_incubator_org' : 'Apache Incubator',
#                    '6_email_mail_mailing' : 'Email Communications',
#                    '9_patch_patches_new' : 'Patches',
#                    '10_commit_committers_committer' : 'Committers/Commits',
#                    '15_license_copyright_notice' : 'Copyright Notice',
#                    '17_jira_browse_jiras' : 'JIRA',
#                    '21_url_website_page' : 'Project Webpage', #
#                    '22_asf_icla_projects' : 'Apache Foundation/Contributor License',
#                    '23_date_day_week' : 'Schedules/Events',
#                    '26_vote_votes_binding' : 'General Voting',
#                    '27_svn_subversion_repos' : 'Subversion (SVN)',
#                    '30_report_developed_reports' : 'Incubator Reporting',
#                    '35_configuration_config_configure' : 'Project Configuration',
#                    '51_delete_script_remove' : 'Deletion/Removal',
#                    '62_pmc_ppmc_members' : 'Podling Project Management Committee (PPMC)',
#                    '80_ipmc_votes_pmc' : 'Incubator Project Management Committee (IPMC)',
#                    '83_artifacts_artifact_nexus' : 'Artifact Management',
#                    '84_request_client_response' : 'Request Handling',
#                    '91_graduation_address_important' : 'Graduation Requirements/Maturity Model',
#                    '107_resource_resources_persistent' : 'Resource Object Management',
#                    '113_hours_vote_open' : 'Voting Protocol/Timeline'}



import os, sys
import torch
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import random
np.random.seed(0)
random.seed(0)

import ast
import itertools
from tqdm import tqdm
tqdm.pandas()
import itertools
import json


import math
from scipy.spatial import distance
from scipy.signal import find_peaks
from scipy.stats import chi2_contingency
from scipy.stats.stats import pearsonr

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.patches as mpatches
plt.rcParams["figure.figsize"] = (11.7,8.27)
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter
from statistics import mode

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from group_lasso import GroupLasso, LogisticGroupLasso
from group_lasso.utils import extract_ohe_groups
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#percentile based outlier removal, not used
def rem_outlier(df,resp):
  init_shape = df.shape[0]
  for col in df.columns:
    if col != resp:
      Q1 = df[col].quantile(0.01)
      Q3 = df[col].quantile(0.99)
      IQR = Q3 - Q1
      print(init_shape)
      outlier = df[((df[col] < Q1 )|(df[col] > Q3))]
      df = df[~((df[col] < Q1 )|(df[col] > Q3))]
  print(df.shape[0])
  print('N lost:', 1-(init_shape/df.shape[0]))

  return df


#merge performance data and semantic measurements along incubation month and project

proj_var = pd.read_csv("SE_metrics.csv")
#rename columns in performance metrics
proj_var = proj_var.rename(columns={'project': 'proj_name','incubation_month':'month'})
#enforce lower case
proj_var['proj_name'] = proj_var['proj_name'].apply(lambda x : x.lower())
#align months between emails anf perf data
proj_var['month'] = proj_var['month'].apply(lambda x : x - 1)
#numeric coding of status
proj_var['status'] = proj_var['status'].apply(lambda x : 1 if x =='graduated' else 0 )
#subset covariates of interest
proj_var = proj_var[['proj_name','month','committers','commits','code','status','devs']]
proj_var.to_csv('data_reduced.csv')

#load semantic metrics
#load posthoc_analysis.csv for secondary analysis
all_data = pd.read_csv('primary_analysis.csv');print(all_data.columns)
#subset variables of interest
all_data = all_data[['cos_score','topic','project_name', 'month','parsed_clauses','sender_email','best_match','date','group_id']]
#rename columns
all_data = all_data.rename(columns={'project_name': 'proj_name'})

#remove the 6 git projects with no commit history
git_missing = ['cayenne','cxf','jena','servicemix','openmeetings','odftoolkit','brooklyn']
all_data = all_data[~all_data['proj_name'].isin(git_missing)]

#check whether both data have same sets of projects
s_data = set(proj_var['proj_name'].to_list())
l_data = set(all_data['proj_name'].to_list())
print(l_data.difference(s_data),len(l_data),len(s_data))

#merge datasets
merged_data = pd.merge(proj_var, all_data, on=['proj_name','month'], how='left')
merged_data.dropna(inplace=True)
print('Total routines:',merged_data.shape[0],all_data.shape[0])

#tally measured observatioins in graduated vs retired
print('graduated ',len(merged_data[merged_data['status']==1]['proj_name'].unique()), merged_data[merged_data['status']==1].shape[0])
print('retired ', len(merged_data[merged_data['status']==0]['proj_name'].unique()), merged_data[merged_data['status']==0].shape[0])

merged_data.to_csv('regression.csv',index=False)


# RQ1
fig, axes = plt.subplots(figsize=(12,15), nrows=1,ncols=2, sharey=True, frameon=True, width_ratios=[1, 1])
fig.tight_layout(pad=1, w_pad=1, h_pad=1.0)
ax1,ax2= axes[0],axes[1]
#create topic_map to assign inferred topics to topic number
#load cluster_rules_posthoc.csv for posthoc analysis
ground_rules = pd.read_csv('cluster_rules.csv')
topic_map = ground_rules[['topic','label']].drop_duplicates(['topic','label'])
topic_map = pd.Series(topic_map.topic.values,index=topic_map.label).to_dict()
topic_map = {k : topic_assigned[v] for k,v in topic_map.items()}
print(topic_map)
n_topics = len(topic_map)

#assign inferred topics
topic_data = pd.read_csv('regression.csv')
topic_data['group_id'] = topic_data['group_id'].apply(lambda x : topic_map[x] )
topic_data = topic_data.dropna(subset=['group_id'])
ground_rules['topic'] = ground_rules['label'].apply(lambda x : topic_map[x])

#normalized densities of ASFI policy
b = ground_rules['topic']
hist = Counter(b); hist = {k:v/b.shape[0] for k,v in hist.items()};print(list(hist.values()))
hist = {k: v for k, v in sorted(hist.items(), key=lambda item: (item[1],item[0]))}
#list of topics for plot label
topics = list(hist.keys())

#normalized densities of project activity
p = topic_data['group_id']; print(mode(p))
histnorm = Counter(p); histnorm = {k:v/p.shape[0] for k,v in histnorm.items()}
histnorm = {elem:histnorm[elem] for elem in topics}

print('Correlation of policy extent and topic wise governed activity: ', pearsonr(list(hist.values()),list(histnorm.values())))
print(list(zip(list(hist.values()),list(histnorm.values()))))


#sort governance topics by regulation (For RQ3)
low, mid, high = ([k  for k,v in hist.items() if v <= np.percentile(list(hist.values()),x)] for x in [33,66,100])
high = [elem for elem in high if elem not in mid]; mid = [elem for elem in mid if elem not in low]
regmap = {'high' : high,'mid' : mid, 'low' : low}


#plot density of policies
ax1.barh(topics, hist.values(),color='royalblue')
ax1.set_xticks(np.arange(0,.15,0.05))
ax1.tick_params(axis='both', which='major', labelsize=16)
ax1.set_title('Topic Wise Normalized \n ASFI Policy Extent', fontweight='bold', fontsize="16")
blue_patch = mpatches.Patch(color='royalblue', label='ASF Policies')
# ax1.legend(handles=[blue_patch],loc='lower right',facecolor='white', fontsize="16") #13EAC9

#plot densities of governance
ax2.barh(topics, histnorm.values(),color='orange')
ax1.set_xticks(np.arange(0,.15,0.05))
ax2.tick_params(axis='both', which='major', labelsize=16)
ax2.set_title('Topic Wise Normalized \n Govenered Activity', fontweight='bold', fontsize="16")
orange_patch = mpatches.Patch(color='orange', label='Routines')
plt.legend(handles=[blue_patch,orange_patch], ncol = 2, loc='lower right',facecolor='white', fontsize="16",bbox_to_anchor=(1,-0.1))
plt.draw()
plt.savefig('topic_email_policy.pdf', format="pdf", bbox_inches="tight",dpi=600)

#RQ2
fig, axes = plt.subplots(figsize=(12,15), nrows=1,ncols=2, sharey=True, frameon=True, width_ratios=[1, 1])
fig.tight_layout(pad=1, w_pad=1, h_pad=1.0)
ax1,ax2= axes[0],axes[1]
print(topic_map)
topics = list(hist.keys())

#Load regression data and assign inferred topics
topic_data = pd.read_csv('regression.csv')
topic_data['group_id'] = topic_data['group_id'].apply(lambda x : topic_map[x] )
topic_data = topic_data.dropna(subset=['group_id'])

#boxplot of compliance
mean_scores, boxcomp = [], pd.DataFrame()

for topic in topics:
  subtopic = topic_data[topic_data['group_id']==topic]['cos_score']
  mean_scores.append(subtopic.mean())
  boxcomp[topic] = subtopic.sample(1000,random_state=0).values

print('Correlation of regulation and mean score: ', pearsonr(list(hist.values()),mean_scores))
#plot density of policies
ax1.barh(topics, hist.values(),color='royalblue')
ax1.tick_params(axis='both', which='major', labelsize=16)
ax1.set_title('Topic Wise Normalized \n ASFI Policy Extent', fontweight='bold', fontsize="15")
blue_patch = mpatches.Patch(color='royalblue', label='ASF Policies')

#plot corresponding boxplot
ax2.set_title('Topic Wise Distribution \n of Internalization', fontweight='bold', fontsize="15")
# ax2.set_xlabel('Compliance Scores')
ax2.set_xticks([0.5,1])
ax2.tick_params(axis='both', which='major', labelsize=16)
boxcomp.boxplot(ax=ax2, vert=False, positions = np.array(range(n_topics)), showmeans=True,
                flierprops={'marker': '.', 'markersize': 1}, medianprops={'color':'red',"linewidth": 2})

plt.yticks(np.array(range(n_topics)),labels=topics)
plt.legend(handles=[blue_patch],loc='lower right',facecolor='white', fontsize="16",bbox_to_anchor=(1,-0.07))
# yax = ax2.get_yaxis()

# yax.set_tick_params(pad=4)
plt.draw()

plt.savefig('topic_scores.pdf', format="pdf", bbox_inches="tight",dpi=600)
plt.show()


#graduation/retirement RQ3
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.preprocessing import StandardScaler
import seaborn as sns
std_scaler = StandardScaler()

#map topics and create dummies
all_data = pd.read_csv('regression.csv')
all_data['topic'] = all_data['group_id'].apply(lambda x : topic_map[x] )
all_data = all_data.dropna(subset=['topic'])
all_data = pd.get_dummies(all_data, columns=['topic'], prefix='', prefix_sep='')

#define covarites to be averaged on months
cols = ['committers','commits','status','devs']

#predictor dataframe
groups = all_data.groupby(['proj_name'])
H3 = pd.DataFrame()


for proj in all_data['proj_name'].unique():
  #select projecr
  df = groups.get_group(proj)
  # get months in incubation
  incub = df['month'].max()
  #metrics averaged over month
  entry = df.drop_duplicates(subset = ['month'])
  #all average project committers, committs and emails over incubation
  entry = entry.mean(numeric_only=True)[cols]
  #codebase at time of graduation
  entry['codebase'],entry['time'] = df[df['month']==incub]['code'].values[0],incub

  for topic in topics:
    # topic routines from project
    dft = df[df[topic]==1]
    # if topic not present
    if not dft.shape[0]:
      entry[topic+'_count'], entry[topic+'_avg_score'] = 0,np.nan
      continue

    #governance frequency
    entry[topic+'_count'] = dft.shape[0]
    # compliance score (averaged)
    entry[topic+'_avg_score'] = dft['cos_score'].mean()
  H3 = H3.append(entry,ignore_index=True)


# find percent missing in predictors
percent_missing = (H3.isnull().sum()*100)/len(H3)
missing_value_df = pd.DataFrame({'column_name': H3.columns,
                                 'percent_missing': percent_missing})
missing_value_df.sort_values('percent_missing', inplace=True)
missing_value_df.to_csv('analysis.csv',index=False)


# check chi sq by predictor missing and status
from scipy.stats import chi2_contingency
for col in H3.columns:
  if 'count' in col:
    sub = H3[[col,'status']]
    sub[col] = sub[col].apply(lambda x: 1 if x else 0)

#store data for scatter plot (later)
H3.to_csv('scatter_score.csv',index=False)
# remove missing topic for validity testing
inc = pd.read_csv('analysis.csv')
missing = [x.split('_')[0] for x in inc[inc['percent_missing'] > 10]['column_name'].values]; print(missing)


# MICE imputation (sklearn)
imp = IterativeImputer(max_iter=1000, random_state=0)
# donot include outcome to regress imputation
temp = H3['status'];H3.drop('status',axis=1,inplace=True)

l_ = H3.copy(); l_['status'] = temp; l_.to_csv('hypothesis3_unimputed.csv',index=False)
#transform imputation
transcol = H3.columns
H3 = pd.DataFrame(imp.fit_transform(H3),columns = transcol)

l_ = H3.copy(); l_['status'] = temp; l_.to_csv('hypothesis3_imputed.csv',index=False)

checkout = 0
H3['status'] = temp
# log transform covariates and governance activity
cols3 = ['committers','commits','codebase','devs']
from scipy import stats
for col in H3.columns:
  if col in cols3 or '_count' in col:
    H3[col].replace(0,1,inplace=True) #add 1 for log transform
    checkout += 1
    H3[col] = H3[col].apply(lambda x : np.log10(x))

print(checkout)
#store predictors
H3.to_csv('hypothesis3.csv',index=False)

import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
LogisticGroupLasso.LOG_LOSSES = True

X = pd.read_csv('hypothesis3.csv')
orgX = X.copy()
Y = X['status']; print(sum(Y))
X.drop(['status'], axis=1,inplace=True)


# standard scale values for lasso
for col in X.columns: X[col] = std_scaler.fit_transform(X[[col]])

def pred_lasso(X,Y):
  all_pred = X.columns.to_list()
  groups = [-1]*len(all_pred)

  l1_regs = 100
  l1regularisations = np.logspace(-1.5, -3, l1_regs)

  gl = LogisticGroupLasso(
      groups=np.array(groups),
      group_reg=0,
      l1_reg=0,
      random_state=0,
      scale_reg="inverse_group_size",
      n_iter=100,
      subsampling_scheme=1,
      supress_warning=True,
      # warm_start=True,
  )

  var_sec = pd.DataFrame(columns=X.columns)
  kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
  var_select = None
  loss_min = np.inf
  model = None

  for j, l1_reg in enumerate(l1regularisations[::-1]):
      gl.l1_reg = l1_reg
      loss = []
      for _, (train_index, test_index) in enumerate(kf.split(X,Y)):
        X_train,X_test = X.loc[train_index,:],X.loc[test_index,:]
        Y_train,Y_test = Y.loc[train_index],Y.loc[test_index]
        gl.fit(X_train, Y_train)
        Y_hat = gl.predict(X_test)
        loss.append(log_loss(Y_test,Y_hat))

      if np.mean(loss) < loss_min :
        var_select,loss_min,model = gl.sparsity_mask_,np.mean(loss),gl

  w_hat = model.coef_
  coef = model.coef_[:, 1] - model.coef_[:, 0]
  plt.plot(
      coef / np.linalg.norm(coef), ".", label="Estimated weights",
  )

  plt.figure()
  plt.show()
  numask = model.transform([np.arange(X.shape[1])])
  print('Check for lasso:', numask[0],np.sum(np.where(coef,1,0)))
  reddata = pd.concat([X.iloc[:,numask[0]],Y],axis=1)
  return reddata

#Lasso selection
reduced = pred_lasso(X,Y)
reduced.drop('status',axis=1,inplace=True)

from statsmodels.stats import outliers_influence as sm
#Multicollinearity test
while(1):
  vif_data = pd.DataFrame(reduced.columns)
  vif_data["feature"] = reduced.columns
  # calculating VIF for each feature
  vif_data["VIF"] = [sm.variance_inflation_factor(reduced.values, i)
                            for i in range(len(reduced.columns))]
  vif_data.sort_values(by='VIF',inplace=True)
  val,remove = vif_data.VIF.values[-1],vif_data.feature.values[-1]
  print(remove,val)
  if val > 5 : reduced.drop(columns=[remove],inplace=True)
  else : break

#store results for regression
reduced['status'] = Y
reduced.to_csv('grouped_H3.csv',index=False)

#model 1
print("M1")
try: reduced = pd.read_csv('grouped_outliers.csv')
except : reduced = pd.read_csv('grouped_H3.csv')

reduced = reduced.drop(reduced.filter(regex='score').columns, axis=1)
reduced = reduced.drop(reduced.filter(regex='count').columns, axis=1)
print(reduced.columns)
import statsmodels.api as sm
response = reduced['status']
reduced.drop(columns=['status'],inplace=True)
print(reduced.shape[1])

#Logistic GLM fit
regress = sm.add_constant(reduced)
glm_binom = sm.GLM(response, regress, family=sm.families.Binomial())
res = glm_binom.fit()
print(res.summary())
##display coefficients by regulation
coeffs = pd.DataFrame(res.params)
coeffs['p values'] = [np.round(res.pvalues.loc[x],3) for x in list(coeffs.index)]
coeffs = coeffs.rename(columns={0: 'coefficient'})
coeffs['regulation'] = [None]*coeffs.shape[0]
coeffs = coeffs[coeffs['p values'] < 0.05]

for k,v in regmap.items():
  for elem in list(coeffs.index):
    if elem.split('_')[0] in v : coeffs.at[elem, 'regulation'] = k

print(coeffs); coeffs.to_csv('significant.csv')


#Find cooks distance outliers
probs = res.predict()
influence = res.get_influence()
summ_df = influence.summary_frame()
# Set Cook's distance threshold
cook_threshold = 4 / len(reduced)
# Plot influence measures (Cook's distance)
fig = influence.plot_index(y_var="cooks", threshold=cook_threshold)
plt.axhline(y=cook_threshold, ls="--", color='red')
fig.tight_layout(pad=2)
# Filter summary df to Cook's distance values only
diagnosis_df = summ_df[['cooks_d']]



# Append absolute standardized residual values
diagnosis_df['std_resid'] = stats.zscore(res.resid_pearson)
diagnosis_df['std_resid'] = diagnosis_df['std_resid'].apply(lambda x: np.abs(x))

diagnosis_df.sort_values(by=['cooks_d'], ascending=False, inplace=True)

#outliers removed based on cook's d was kept empty, as no D > 1 leverage points were found in analyses
outlier = []

try:
  X = pd.read_csv('grouped_outliers.csv')
except:
  X = pd.read_csv('grouped_H3.csv')
  X.drop(index=outlier,inplace=True);print(X.shape)
  X.to_csv('grouped_outliers.csv',index=False)


## box tidwell for Logistic GLM assumptions
## Add constant to prevent nan/inf during log. 8 is the smallest constant term found to prevent error
reduced += 8
# Add logit transform interaction terms (natural log) for continuous variables e.g.. Age * Log(Age)
for var in reduced.columns:
    reduced[f'{var}:Log_{var}'] = reduced[var].apply(lambda x: x * np.log(x))

X_lt_constant = sm.add_constant(reduced)
logit_results = sm.GLM(response, X_lt_constant, family=sm.families.Binomial()).fit()

# Display summary results
print(logit_results.summary())
print('actual', np.mean(response), 'modeled',np.mean(np.where(probs > .5,1,0)))
output = pd.DataFrame()
output['predicted'],output['actual'] = np.where(probs > .5,1,0),response.values
contigency_grad = pd.crosstab(output['predicted'],output['actual'],margins = False)
print(contigency_grad)
print(chi2_contingency(contigency_grad))
print('f1_score',f1_score(response, np.where(probs > .5,1,0), average='weighted'))
print('accuracy_score',accuracy_score(response, np.where(probs > .5,1,0)))

#model 2

print("M2")
try: reduced = pd.read_csv('grouped_outliers.csv')
except : reduced = pd.read_csv('grouped_H3.csv')

reduced = reduced.drop(reduced.filter(regex='score').columns, axis=1)
# reduced = reduced.drop(reduced.filter(regex='count').columns, axis=1)
print(reduced.columns)
import statsmodels.api as sm
response = reduced['status']
reduced.drop(columns=['status'],inplace=True)
print(reduced.shape[1])

#Logistic GLM fit
regress = sm.add_constant(reduced)
glm_binom = sm.GLM(response, regress, family=sm.families.Binomial())
res = glm_binom.fit()
print(res.summary())
##display coefficients by regulation
coeffs = pd.DataFrame(res.params)
coeffs['p values'] = [np.round(res.pvalues.loc[x],3) for x in list(coeffs.index)]
coeffs = coeffs.rename(columns={0: 'coefficient'})
coeffs['regulation'] = [None]*coeffs.shape[0]
coeffs = coeffs[coeffs['p values'] < 0.05]

for k,v in regmap.items():
  for elem in list(coeffs.index):
    if elem.split('_')[0] in v : coeffs.at[elem, 'regulation'] = k

print(coeffs); coeffs.to_csv('significant.csv')


#Find cooks distance outliers
probs = res.predict()
influence = res.get_influence()
summ_df = influence.summary_frame()
# Set Cook's distance threshold
cook_threshold = 4 / len(reduced)
# Plot influence measures (Cook's distance)
fig = influence.plot_index(y_var="cooks", threshold=cook_threshold)
plt.axhline(y=cook_threshold, ls="--", color='red')
fig.tight_layout(pad=2)
# Filter summary df to Cook's distance values only
diagnosis_df = summ_df[['cooks_d']]



# Append absolute standardized residual values
diagnosis_df['std_resid'] = stats.zscore(res.resid_pearson)
diagnosis_df['std_resid'] = diagnosis_df['std_resid'].apply(lambda x: np.abs(x))

diagnosis_df.sort_values(by=['cooks_d'], ascending=False, inplace=True)
#outliers removed based on cook's d was kept empty, as no D > 1 leverage points were found in analyses
outlier = []

try:
  X = pd.read_csv('grouped_outliers.csv')
except:
  X = pd.read_csv('grouped_H3.csv')
  X.drop(index=outlier,inplace=True);print(X.shape)
  X.to_csv('grouped_outliers.csv',index=False)

reduced += 8
# Add logit transform interaction terms (natural log) for continuous variables e.g.. Age * Log(Age)
for var in reduced.columns:
    reduced[f'{var}:Log_{var}'] = reduced[var].apply(lambda x: x * np.log(x))

X_lt_constant = sm.add_constant(reduced)
logit_results = sm.GLM(response, X_lt_constant, family=sm.families.Binomial()).fit()

# Display summary results
print(logit_results.summary())
print('actual', np.mean(response), 'modeled',np.mean(np.where(probs > .5,1,0)))
output = pd.DataFrame()
output['predicted'],output['actual'] = np.where(probs > .5,1,0),response.values
contigency_grad = pd.crosstab(output['predicted'],output['actual'],margins = False)
print(contigency_grad)
print(chi2_contingency(contigency_grad))
print('f1_score',f1_score(response, np.where(probs > .5,1,0), average='weighted'))
print('accuracy_score',accuracy_score(response, np.where(probs > .5,1,0)))

#model 3

print("M3")
try: reduced = pd.read_csv('grouped_outliers.csv')
except : reduced = pd.read_csv('grouped_H3.csv')

# reduced = reduced.drop(reduced.filter(regex='score').columns, axis=1)
reduced = reduced.drop(reduced.filter(regex='count').columns, axis=1)
print(reduced.columns)
import statsmodels.api as sm
response = reduced['status']
reduced.drop(columns=['status'],inplace=True)
print(reduced.shape[1])

#Logistic GLM fit
regress = sm.add_constant(reduced)
glm_binom = sm.GLM(response, regress, family=sm.families.Binomial())
res = glm_binom.fit()
print(res.summary())
##display coefficients by regulation
coeffs = pd.DataFrame(res.params)
coeffs['p values'] = [np.round(res.pvalues.loc[x],3) for x in list(coeffs.index)]
coeffs = coeffs.rename(columns={0: 'coefficient'})
coeffs['regulation'] = [None]*coeffs.shape[0]
coeffs = coeffs[coeffs['p values'] < 0.05]

for k,v in regmap.items():
  for elem in list(coeffs.index):
    if elem.split('_')[0] in v : coeffs.at[elem, 'regulation'] = k

print(coeffs); coeffs.to_csv('significant.csv')


#Find cooks distance outliers
probs = res.predict()
influence = res.get_influence()
summ_df = influence.summary_frame()
# Set Cook's distance threshold
cook_threshold = 4 / len(reduced)
# Plot influence measures (Cook's distance)
fig = influence.plot_index(y_var="cooks", threshold=cook_threshold)
plt.axhline(y=cook_threshold, ls="--", color='red')
fig.tight_layout(pad=2)
# Filter summary df to Cook's distance values only
diagnosis_df = summ_df[['cooks_d']]



# Append absolute standardized residual values
diagnosis_df['std_resid'] = stats.zscore(res.resid_pearson)
diagnosis_df['std_resid'] = diagnosis_df['std_resid'].apply(lambda x: np.abs(x))

diagnosis_df.sort_values(by=['cooks_d'], ascending=False, inplace=True)

#outliers removed based on cook's d was kept empty, as no D > 1 leverage points were found in analyses
outlier = []

try:
  X = pd.read_csv('grouped_outliers.csv')
except:
  X = pd.read_csv('grouped_H3.csv')
  X.drop(index=outlier,inplace=True)
  X.to_csv('grouped_outliers.csv',index=False)


# box tidwell for Logistic GLM assumptions

reduced += 8
# Add logit transform interaction terms (natural log) for continuous variables e.g.. Age * Log(Age)
for var in reduced.columns:
    reduced[f'{var}:Log_{var}'] = reduced[var].apply(lambda x: x * np.log(x))

X_lt_constant = sm.add_constant(reduced)
logit_results = sm.GLM(response, X_lt_constant, family=sm.families.Binomial()).fit()

# Display summary results
print(logit_results.summary())
print('actual', np.mean(response), 'modeled',np.mean(np.where(probs > .5,1,0)))
output = pd.DataFrame()
output['predicted'],output['actual'] = np.where(probs > .5,1,0),response.values
contigency_grad = pd.crosstab(output['predicted'],output['actual'],margins = False)
print(contigency_grad)
print(chi2_contingency(contigency_grad))
print('f1_score',f1_score(response, np.where(probs > .5,1,0), average='weighted'))
print('accuracy_score',accuracy_score(response, np.where(probs > .5,1,0)))

#model 4

print("M4")
try: reduced = pd.read_csv('grouped_outliers.csv')
except : reduced = pd.read_csv('grouped_H3.csv')

print(reduced.columns)
import statsmodels.api as sm
response = reduced['status']
reduced.drop(columns=['status'],inplace=True)
print(reduced.shape[1])

#Logistic GLM fit
regress = sm.add_constant(reduced)
glm_binom = sm.GLM(response, regress, family=sm.families.Binomial())
res = glm_binom.fit()
print(res.summary())
##display coefficients by regulation
coeffs = pd.DataFrame(res.params)
coeffs['p values'] = [np.round(res.pvalues.loc[x],3) for x in list(coeffs.index)]
coeffs = coeffs.rename(columns={0: 'coefficient'})
coeffs['regulation'] = [None]*coeffs.shape[0]
coeffs = coeffs[coeffs['p values'] < 0.05]

for k,v in regmap.items():
  for elem in list(coeffs.index):
    if elem.split('_')[0] in v : coeffs.at[elem, 'regulation'] = k

print(coeffs); coeffs.to_csv('significant.csv')


#Find cooks distance outliers
probs = res.predict()
influence = res.get_influence()
summ_df = influence.summary_frame()
cook_threshold = 4 / len(reduced)
# Plot influence measures (Cook's distance)
fig = influence.plot_index(y_var="cooks", threshold=cook_threshold)
plt.axhline(y=cook_threshold, ls="--", color='red')
fig.tight_layout(pad=2)
# Filter summary df to Cook's distance values only
diagnosis_df = summ_df[['cooks_d']]

# Set Cook's distance threshold
cook_threshold = 4 / len(reduced)

# Append absolute standardized residual values
diagnosis_df['std_resid'] = stats.zscore(res.resid_pearson)
diagnosis_df['std_resid'] = diagnosis_df['std_resid'].apply(lambda x: np.abs(x))

diagnosis_df.sort_values(by=['cooks_d'], ascending=False, inplace=True)

#outliers removed based on cook's d was kept empty, as no D > 1 leverage points were found in analyses
outlier = []

try:
  X = pd.read_csv('grouped_H3.csv')
  X.drop(index=outlier,inplace=True);print(X.shape)
  X.to_csv('grouped_outliers.csv',index=False)
except:pass


# box tidwell for Logistic GLM assumptions

reduced += 8
# Add logit transform interaction terms (natural log) for continuous variables e.g.. Age * Log(Age)
for var in reduced.columns:
    reduced[f'{var}:Log_{var}'] = reduced[var].apply(lambda x: x * np.log(x))

X_lt_constant = sm.add_constant(reduced)
logit_results = sm.GLM(response, X_lt_constant, family=sm.families.Binomial()).fit()

# Display summary results
print(logit_results.summary())
print('actual', np.mean(response), 'modeled',np.mean(np.where(probs > .5,1,0)))
output = pd.DataFrame()
output['predicted'],output['actual'] = np.where(probs > .5,1,0),response.values
contigency_grad = pd.crosstab(output['predicted'],output['actual'],margins = False)
print(contigency_grad)
print(chi2_contingency(contigency_grad))
print('f1_score',f1_score(response, np.where(probs > .5,1,0), average='weighted'))
print('accuracy_score',accuracy_score(response, np.where(probs > .5,1,0)))


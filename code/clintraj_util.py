import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2_contingency
from statsmodels.formula.api import ols

def associate_with_categorical_var(categorical_var,variable,var_type,score_threshold=0.7,pvalue_threshold_parameter=1e-10,verbose=True,Negative_Enrichment=False,Minimal_number_of_points=5,produce_plot=False):
    # compute an association of a categorical variable categorical_var with another variableof type var_type
    # (chi-square : binary, ordinal and categorical and ANOVA for continuous)
    if var_type=='BINARY' or var_type=='CATEGORICAL':
        d = {'CATEGORICAL': categorical_var, 'VAR': variable} 
        df = pd.DataFrame(data=d)
        table = pd.crosstab(df['CATEGORICAL'], df['VAR'])
        var_vals = list(table.columns)
        categorical_vals = list(table.index)
        #print(categorical_vals)
        #print(var_vals)
        stat, p, dof, expected = chi2_contingency(table)
        if verbose:
            display(table)
            print(expected)
        devs = (table-expected)/(table+expected)
        devs = devs.to_numpy()
        table_array = table.to_numpy()
        list_of_associations = []
        for i in range(devs.shape[0]):
            for j in range(devs.shape[1]):
                if devs[i,j]>score_threshold and table_array[i,j]>=Minimal_number_of_points and not Negative_Enrichment:
                    if verbose:
                        print(str(categorical_vals[i])+'\t'+str(var_vals[j])+'\t{}'.format(devs[i,j]))
                    list_of_associations.append((categorical_vals[i],var_vals[j],devs[i,j]))
                if devs[i,j]<-score_threshold and expected[i,j]>=Minimal_number_of_points and Negative_Enrichment:
                    if verbose:
                        print(str(categorical_vals[i])+'\t'+str(var_vals[j])+'\t{}'.format(devs[i,j]))
                    list_of_associations.append((categorical_vals[i],var_vals[j],devs[i,j]))
    if var_type=='CONTINUOUS' or var_type=='ORDINAL':
        d = {'CATEGORICAL': categorical_var, 'VAR': variable} 
        df = pd.DataFrame(data=d)
        categorical_vals = np.unique(categorical_var)
        #df = df.get_dummies(['CATEGORICAL'])
        #results = gls('VAR ~ C(CATEGORICAL)', data=df,hasconst=True).fit()
        results = ols('VAR ~ C(CATEGORICAL)', data=df).fit()
        if verbose:
            display(results.summary())
        aov_table = sm.stats.anova_lm(results, typ=2)
        if verbose:
            display(aov_table)
        pvals = np.asarray([v for v in results.pvalues])
        scores = np.asarray([v for v in results.params])
        if verbose:
            print(categorical_vals)
            print('P-values',pvals)
            print('Scores:',scores)
        list_of_associations = []
        for i,p in enumerate(pvals):
            if p<pvalue_threshold_parameter:
                if scores[i]>score_threshold and not Negative_Enrichment:
                    list_of_associations.append((categorical_vals[i],'CONTINUOUS',scores[i]))
                if scores[i]<-score_threshold and Negative_Enrichment:
                    list_of_associations.append((categorical_vals[i],'CONTINUOUS',scores[i]))
        if produce_plot:
            my_dict = {}
            unique_vals = np.unique(categorical_var)
            for val in unique_vals:
                my_dict[val] = vals[np.where(categorical_var==val)]
            fig, ax = plt.subplots()
            ax.boxplot(my_dict.values())
            ax.set_xticklabels(my_dict.keys())      

        stat = aov_table.F[0]
        p = aov_table['PR(>F)'][0]
    return list_of_associations,p,stat
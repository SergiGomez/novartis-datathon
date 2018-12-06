
catVarsDict = {'Country' : 'LabelEncoder',
               'Brand_Group': 'LabelEncoder',
               'Cluster': 'LabelEncoder'}

feats_to_add = [
                'trend',
                'yearly',
                'yearly_lower',
                'yearly_upper',
                'trend_lower',
                'trend_upper',

                 'mean_target_Cluster_sales1',
                 'mean_target_Country_sales1',
                 'mean_target_Brand_Group_sales1',
                 'mean_target_Month_sales1',
                 'mean_target_Cluster_sales2',
                 'mean_target_Country_sales2',
                 'mean_target_Brand_Group_sales2'
                  'mean_target_Month_sales2',
                  'mean_target_Cluster_inv1',
                 # 'mean_target_Country_inv1',
                  'mean_target_Year_inv1',
                 # 'mean_target_Month_inv1',
                 # 'mean_target_Cluster_inv2',
                 # 'mean_target_Country_inv2',
                 # 'mean_target_Brand_Group_inv2',
                  'mean_target_Year_inv2',
                 # 'mean_target_Month_inv2',
                 # 'mean_target_Cluster_inv3',
                  'mean_target_Year_inv3',
                 # 'mean_target_Month_inv3',
                 # 'mean_target_Cluster_inv4',
                 # 'mean_target_Brand_Group_inv4',
                  'mean_target_Year_inv4',
                 # 'mean_target_Month_inv4',
                 # 'mean_target_Cluster_inv6',
                  'mean_target_Year_inv6',
                 # 'mean_target_Month_inv6',

                 'ratio_sales_inv_1',
                 'ratio_sales_inv_2',
                 'ratio_sales_inv_3'

                 'lag12'
                ]

primary_keys = ['Cluster', 'Brand_Group', 'Country']
#primary_keys = ['Cluster', 'Brand_Group']

feats_join =  primary_keys + ['date']

time_vars = ['year', 'month']

vars_model_final = primary_keys + time_vars + feats_to_add

params = {

    'walk-forward-cv' : {
        "n_splits" : 5 # sets the number of folds for cross-validation
    },

    'lgb' : {
        "objective" : "regression",
        "metric" : "mape",
        "num_leaves" : 200,
        "n_estimators" : 1000,
        "learning_rate" : 0.017,
        "bagging_fraction" : 0.6,
        "feature_fraction" : 0.6,
        "bagging_frequency" : 6,
        "bagging_seed" : 42,
        "verbosity" : 1,
        'max_bin': 2000,
        "seed": 42
        },

    'vars_model' : vars_model_final

}

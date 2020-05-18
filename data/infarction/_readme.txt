Files and their provenances:

-- initial_table.csv - initial csv table downloaded from the https://leicester.figshare.com/articles/Myocardial_infarction_complications_Database/12045261/1
    -- infarctus_na.txt - converted to tab-delimited, replaced empty values to N/A
        -- var_types.txt - types of the variables (CONTINUOUS, BINARY, ORDINAL, CATEGORICAL)
        -- infarctus_na_v30s20.txt - removed columns containing more than 30 N/As, and then rows containing more than 20 N/As
        -- all.txt - same table, just renamed
            -- all_complete.txt - rows of all.txt without missing values
            -- all_dummies.txt - all.txt with dummy coding of the categorical variable LET_IS
                -- all_dummies_complete.txt - rows of all_dummies.txt without missing values
                -- all_dummies_imp_SVDcomplete.txt - imputed matrix using SVDcomplete method
                -- all_dummies_norm_info.txt - univariate quantification information for the variables of all_dummies.txt
                -- all_dummies_q_imp_SVDcomplete.txt - quantified imputed matrix using SVDcomplete method
        -- indeps.txt - only 'independent' variables (not complications)
		-- indeps_complete.txt - rows of indeps_dummies.txt without missing values
		-- indeps_imp_SVDcomplete.txt - imputed matrix using SVDcomplete method
		-- indeps_norm_info.txt - univariate quantification information for the variables of all_dummies.txt
		-- indeps_q_imp_SVDcomplete.txt - quantified imputed matrix using SVDcomplete method
        -- deps.txt - only 'dependent' variables (complications)
            -- deps_dummies.txt - dummy coding of the categorical variable LET_IS
                -- deps_dummies_complete.txt - rows of deps_dummies.txt without missing values 
                -- deps_dummies_imp_SVDcomplete.txt - imputed matrix using SVDcomplete method
                -- deps_dummies_norm_info.txt - univariate quantification information for the variables of all_dummies.txt
                -- deps_dummies_q_imp_SVDcomplete.txt - quantified imputed matrix using SVDcomplete method


* SVDcomplete method for missing value imputation:
	0) k parameter is chosen (number of principal components) - intrinsic linear dimensionality seems to be not a bad choice
	1) SVD (PCA) is computed for the complete rows of the dataset, for k components
	2) Vectors with missing values are projected onto the k principal components (closest point of the manifold, the scalar product is adapted accordingly)
	3) The imputed value is read from the projection

* SVDmissing method for missing value imputation:
	0) k parameter is chosen (number of principal components) - intrinsic linear dimensionality seems to be not a bad choice	
	1) A special version of SVD (PCA) is computed for the complete dataset, including missing values, for k components
	2) Vectors with missing values are projected onto the k principal components (closest point of the manifold, the scalar product is adapted accordingly)
	3) The imputed value is read from the projection


from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, recall_score, precision_score, f1_score, \
log_loss, auc, roc_auc_score, roc_curve
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Creating the class that will store my data 

class Model():
    """
    Container class to store classification model attributes
    """
    
    model_list = []
    model_df = pd.DataFrame(columns=['name','train_accuracy','train_prec','train_recall','train_f1','train_logloss',\
                                     'test_accuracy','test_prec','test_recall','test_f1','test_logloss', "AUC"])
    
    def __init__(self, name, model, X_train, X_test, y_train, y_test, threshold=.5):
        self.name = name
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.threshold = threshold
        
        # Collection of training attributes
        # f1, recall, precision add "_macro" for multi-class
        self.train_results = cross_validate(self.model, self.X_train, self.y_train, scoring=[
            'precision', 'accuracy', 'recall', 'f1', 'neg_log_loss'], n_jobs=4, verbose=1)
        # Train metrics
        self.train_acc = np.mean(self.train_results['test_accuracy'])
        self.train_prec = np.mean(self.train_results['test_precision']) # add "_macro" for multi-class
        self.train_rec = np.mean(self.train_results['test_recall'])  # add "_macro" for multi-class
        self.train_f1 = np.mean(self.train_results['test_f1'])  # add "_macro" for multi-class
        self.train_logloss = -np.mean(self.train_results['test_neg_log_loss'])
        
        # Test metrics
        self.y_pred_proba = self.model.predict_proba(self.X_test)
        
        # Thresholds!!! 
        if self.threshold == .5:
            self.y_pred = self.model.predict(self.X_test)
        
        else:
            self.y_pred = (self.model.predict_proba(self.X_test)[:, 1] >= self.threshold).astype(int)
        
        self.test_score = model.score(self.X_test, self.y_test)
        # average = "macro" for multi-class
        self.test_recall = recall_score(self.y_test, self.y_pred, average='binary', zero_division=0)
        # average = "macro" for multi-class
        self.test_prec = precision_score(self.y_test, self.y_pred, average='binary', zero_division=0)
        self.test_log_loss = log_loss(self.y_test, self.y_pred_proba)
        # average = "macro" for multi-class
        self.test_f1 = f1_score(self.y_test, self.y_pred, average='binary', zero_division=0)
        
        # AUC metrics -> Remove when we get to multi-class
        self.auc = roc_auc_score(self.y_test, self.y_pred_proba[:,1])
        
        # Add model object to the class data container for access within the notebook
        Model.model_list.append(self)
        
        # Dictionary containing all of the metrics to add to the dataframe
        self.attributes = {'name':self.name, 'train_accuracy':self.train_acc, "train_prec": self.train_prec,
                           "train_recall": self.train_rec, "train_f1": self.train_f1, \
                           "train_logloss": self.train_logloss, \
                          'test_accuracy':self.test_score, "test_prec": self.test_prec,
                           "test_recall": self.test_recall, "test_f1": self.test_f1, \
                           "test_logloss": self.test_log_loss, "AUC":self.auc}
        
        # Add the metrics to the class dataframe
        Model.model_df.loc[len(Model.model_df)] = self.attributes
        
        
        
    # Roc Curve plot method -> needs to be removed for multi-class
    def roc_curve(self):
        # Create the plot
        sns.set_style("dark")
        fig, ax = plt.subplots(figsize=(6,6))
        # get the predict_proba values
        y_hat_hd = self.y_pred_proba[:, 1]

        # Get the FPR and TPR data
        fpr, tpr, thresholds = roc_curve(self.y_test, y_hat_hd)
        # Plot the actual graph
        ax.plot(fpr, tpr, label=f"{self.name} | AUC: {self.auc:.2f})")
        # Y-axis
        ax.set_yticks([0,.2,.4,.6,.8,1])
        ax.set_yticklabels([0,20,40,60,80,100])
        ax.set_ylabel("True Positive %")
        # X-axis
        ax.set_xticks([0,.2,.4,.6,.8,1])
        ax.set_xticklabels([0,20,40,60,80,100])
        ax.set_xlabel("False Positive %")
        ax.set_title(f"{self.name} ROC Curve", fontsize=20)
        plt.grid(False);
        
        
    def compare_roc_curve(self):
        sns.set_style("dark")
        # Color Palette
        colors = sns.color_palette("Paired", n_colors=30)
        # Create the plot
        fig, ax = plt.subplots(figsize=(7,7))
        for i in range(len(Model.model_list)):
            # get the predict_proba values
            y_hat_hd = Model.model_list[i].y_pred_proba[:, 1]

            # Get the FPR and TPR data
            fpr, tpr, thresholds = roc_curve(Model.model_list[i].y_test, y_hat_hd)
            # Plot the actual graph
            ax.plot(fpr, tpr, color=colors[i], label=f'{Model.model_list[i].name} | AUC: {Model.model_list[i].auc:.2f})')
        
        ax.set_title(f"Comparison of ROC Curve")
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
        plt.grid(False);
    
    
    def confusion_matrix(self):
        sns.set_style('white')
        # Confusion Matrix Plot
        fig, ax = plt.subplots(figsize=(6,6))
        self.cm = ConfusionMatrixDisplay.from_predictions(y_true=self.y_test, y_pred=self.y_pred, ax=ax)
        plt.title(f'Confusion Matrix for "{self.name}" Test Data');
    
    
    def __str__(self):
      return f"Model name: {self.name}"





# Functions to aid in collection

# Extract Feature Importance from Tree-based Models

def features_from_trees(model_class, number_of_features=5):
    """
    Extracts and zips feature importances with their feature names for tree-based models like Random Forest,
    Extra Trees, and XGBoost
    
    Parameters
    ------------
    model_class: variable name of class Model instance | 
        i.e. rfc = Model("Random Forest Class") ->   <rfc> would be the desired model_class argument
    
    number_of_features: int| number of features desired as output of print() side-effect, no bearing on return value
    
    Returns:
    -----------
    sorted list: a list of tuples sorted by the "feature_importance" value (feature_name, importance)
    """

    # Extracting feature importances and adding them to a dataframe to contain them for each model

    features = list(model_class.model.get_params()["ct"].get_feature_names_out())
    features_list = [i.replace("num_pipe__", "").replace("cat_pipe__","") for i in features]

    imp_feats = model_class.model.get_params()['model'].feature_importances_

    imp_list = list(zip(features_list, imp_feats))
    imp_dict = dict(imp_list)

    idx = len(Importance.df)
    Importance(imp_dict)
    Importance.df.rename(index={idx:model_class.name}, inplace=True)
    

    print(f"Top {number_of_features} Feature Importances")
    for i in sorted(imp_list, key=lambda x: x[1], reverse=True)[:number_of_features]:
        print(i)
    
    return sorted(imp_list, key=lambda x: x[1], reverse=True)
        


# Extracts the coefficients from Logistic Regression models and pairs them with their features

def coefs_from_lr(model_class, number_of_coefs=5):
    
    """
    Extracts and zips feature coefficients with their feature names for parametric models like Logistic Regression
    
    Parameters
    ------------
    model_class: variable name of class Model instance | 
        i.e. logreg = Model("Logistic Regression") ->   <logreg> would be the desired model_class argument
    
    number_of_features: int| number of features desired as output of print() side-effect, no bearing on return value
    
    Returns:
    -----------
    sorted list: a list of tuples sorted by the absolute value of the "coefficient" value 
        (feature_name, coefficient)
    """
    
    features = model_class.model.named_steps["ct"].get_feature_names_out()
    features_list = [i.replace("num_pipe__", "").replace("cat_pipe__","") for i in features]
    
    coef_feats = model_class.model.get_params()['model'].coef_[0]
    
    imp_list = list(zip(features_list, coef_feats))
    imp_dict = dict(imp_list)
    
    idx = len(Importance.df)
    Importance(imp_dict)
    Importance.df.rename(index={idx:model_class.name}, inplace=True)

    print(f"Top {number_of_coefs} Feature Coefficients by Absolute Value")
    for i in sorted(imp_list, key=lambda x: np.abs(x[1]), reverse=True)[:number_of_coefs]:
        print(i)
    
    return sorted(imp_list, key=lambda x: np.abs(x[1]), reverse=True)


def get_polarized_magnitudes(num_of_features, model_name):
    """
    Function to display "num_of_features" number of largest and smallest coefficients or feature importances
    
    Parameters:
    -------------
    num_of_features: int | number of large/small coefficients/features returned
    model_name: str | string name of model stored in Model.name for the custom model class
    
    Returns:
    -------------
    combined: concatenated df with one row as a partition between the largest and smallest values
    """
    # Sort and select top features
    global Importance
    top_features_high = Importance.df.T.sort_values(by=model_name, ascending=False)[:num_of_features]
    top_features_low = Importance.df.T.sort_values(by=model_name, ascending=True)[:num_of_features]
    
    # Create an empty DataFrame with NaN values
    empty_row = pd.DataFrame([[99999] * len(top_features_high.columns)], columns=top_features_high.columns)
    empty_row.rename(index={0:"PARTITION Large Above|Small Below"}, inplace=True)
    # Concatenate DataFrames with an empty row in between
    combined = pd.concat([top_features_high, empty_row, top_features_low], axis=0)
    
    return combined


def get_largest_magnitudes(num_of_features, extracted_features_list):
    """
    Function to display "num_of_features" number of largest magnitude coefficients or feature importances
    
    Parameters:
    -------------
    num_of_features: int | number of large/small coefficients/features returned
    model_name: str | string name of model stored in Model.name for the custom model class
    
    Returns:
    -------------
    combined: concatenated df with one row as a partition between the largest and smallest values
    """
    # Create DataFrame from list of tuples
    data = pd.DataFrame(extracted_features_list[:num_of_features])
    data.rename(columns={0:"Feature Name", 1:"Coef/Importance"}, inplace=True)
    return data
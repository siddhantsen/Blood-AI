from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import FTTransformerConfig
import pandas as pd
import torch
import argparse
import optuna

def main():
    print("hello")
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='PyTorch Tabular Model Training')
    parser.add_argument('--train', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--test', type=str, required=True, help='Name of target column')
    parser.add_argument('--val', type=str, required=True,  help='Number of training epochs')
    parser.add_argument('--output', type=str, default='model_save', help='Directory to save trained model')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size for training')
    args = parser.parse_args()
    
    labs = {
        "51221": "Hematocrit",
        "51265": "Platelet Count",
        "50912": "Creatinine",
        "50971": "Potassium",
        "51222": "Hemoglobin",
        "51301": "White Blood Cells",
        "51249": "MCHC",
        "51279": "Red Blood Cells",
        "51250": "MCV",
        "51248": "MCH",
        "51277": "RDW",
        "51006": "Urea Nitrogen",
        "50983": "Sodium",
        "50902": "Chloride",
        "50882": "Bicarbonate",
        "50868": "Anion Gap",
        "50931": "Glucose",
        "50960": "Magnesium",
        "50893": "Calcium, Total",
        "50970": "Phosphate",
        "51237": "INR(PT)",
        "51274": "PT",
        "51275": "PTT",
        "51146": "Basophils",
        "51256": "Neutrophils",
        "51254": "Monocytes",
        "51200": "Eosinophils",
        "51244": "Lymphocytes",
        "52172": "RDW-SD",
        "50934": "H",
        "51678": "L",
        "50947": "I",
        "50861": "Alanine Aminotransferase (ALT)",
        "50878": "Asparate Aminotransferase (AST)",
        "50813": "Lactate",
        "50863": "Alkaline Phosphatase",
        "50885": "Bilirubin, Total",
        "50820": "pH",
        "50862": "Albumin",
        "50802": "Base Excess",
        "50821": "pO2",
        "50804": "Calculated Total CO2",
        "50818": "pCO2",
        "52075": "Absolute Neutrophil Count",
        "52073": "Absolute Eosinophil Count",
        "52074": "Absolute Monocyte Count",
        "52069": "Absolute Basophil Count",
        "51133": "Absolute Lymphocyte Count",
        "50910": "Creatine Kinase (CK)",
        "52135": "Immature Granulocytes"
    }
    labs_reversed = {value: key for key, value in labs.items()}
    
    df_train = pd.read_excel(args.train, index_col = "Unnamed: 0")
    df_val = pd.read_excel(args.val, index_col = "Unnamed: 0")
    df_test = pd.read_excel(args.test, index_col = "Unnamed: 0")
    
    total_feats = ['Hematocrit',
     'PTT',
     'Asparate Aminotransferase (AST)',
     'Chloride',
     'White Blood Cells',
     'Potassium',
     'Calcium, Total',
     'Phosphate',
     'Monocytes',
     'Eosinophils',
     'Urea Nitrogen',
     'pH',
     'pCO2']
    
    encode = lambda x: [labs_reversed[i] for i in x]
    decode = lambda x: [labs[i] for i in x]
    
    cols = decode(df_train.columns.to_list())
    targets = list(set(cols) - set(total_feats))
    
    data_config = DataConfig(
        target=encode(targets),
        continuous_cols=encode(total_feats),
    )
    trainer_config = TrainerConfig(
        auto_lr_find=True,
        batch_size=args.batch_size,
        accelerator="gpu",
        max_epochs=99
    )
    optimizer_config = OptimizerConfig()

    def objective(trial):
        params={
        # Define hyperparameters to tune
            'num_heads': trial.suggest_categorical("num_heads", [2, 4, 8, 16]),
            'num_attn_blocks': trial.suggest_categorical('num_attn_blocks',[2, 3, 4, 6, 8]),
            'input_embed_dim': trial.suggest_categorical("input_embed_dim", [32, 64, 128, 256, 512])
        }
    
        model_config = FTTransformerConfig(
            num_heads=params['num_heads'],          # Number of attention heads
            num_attn_blocks=params['num_attn_blocks'],  # Number of transformer blocks
            input_embed_dim=params['input_embed_dim'], 
            embedding_dropout=0.1,            # Dropout for feature embeddings
            attn_dropout=0.1,            # Dropout for attention layers
            ff_dropout=0.1,                  # Dropout in feed-forward network
            task="regression",                # or "regression"
            target_range=[(0,1)]*len(targets)
        )
        
        # Initialize and train the model
        tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config
        )
        tabular_model.fit(train=df_train, validation=df_val)

        Y_test =  torch.tensor(df_test[encode(targets)].values).type(torch.float).cuda()
        
        def compute_mse_per_covariate(predictions, targets):
            # Ensure predictions and targets are the same shape
            assert predictions.shape == targets.shape, "Shapes of predictions and targets must match"
            
            # Compute squared error per covariate and average over the batch (dim=0)
            mse_per_covariate = torch.mean((predictions - targets) ** 2, dim=0)
            
            return mse_per_covariate  # Returns a tensor of shape (15,)
        
        y_pred = tabular_model.predict(df_test)
        
        mse_per_covariate = compute_mse_per_covariate(torch.tensor(y_pred.values).type(torch.float).cuda(), Y_test)
        mean = mse_per_covariate.mean()
    
        return mean
    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials= 10)
    
    # Print the best hyperparameters
    print('Best trial:')
    trial = study.best_trial
    print(f'  MSE: {trial.value}')
    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')
        with open("best_trail.txt","a") as f:
            f.write(f'{key}: {value}\n')
    
if __name__ == "__main__":
    main()
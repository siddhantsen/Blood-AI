from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import NodeConfig
import pandas as pd
import torch
import argparse

def main():
    print("hello")
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='PyTorch Tabular Model Training')
    parser.add_argument('--train', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--test', type=str, required=True, help='Name of target column')
    parser.add_argument('--val', type=str, required=True,  help='Number of training epochs')
    parser.add_argument('--layers', type=int, required=True, help='Batch size for training')
    parser.add_argument('--trees', type=int, required=True, help='Batch size for training')
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
        accelerator="gpu"
    )
    optimizer_config = OptimizerConfig()
    
    model_config = NodeConfig(
        num_layers=args.layers,
        num_trees=args.trees,
        task="regression",  # or "regression"
        head="LinearHead",      # Using LinearHead with sigmoid
        head_config={
            "layers": None,    # No additional layers
            "activation": "Sigmoid"  # Sigmoid activation
        },
        #data_aware_init_batch_size=1000,
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
    
    mse_per_covariate = compute_mse_per_covariate(torch.tensor(y_pred.values).type(torch.float).cuda(),Y_test)
        # Convert MSE tensor to numpy and pair with column names
    print(f"Overall MSE: {mse_per_covariate.mean()}")
    with torch.no_grad(): 
        mse_per_covariate_np = mse_per_covariate.cpu().numpy()  # If using GPU: .cpu().numpy()
    
    # Display as a DataFrame for better readability
    mse_df = pd.DataFrame({
        'Covariate': targets,
        'MSE': mse_per_covariate_np
    })
    
    order = ['Albumin',
     'Alkaline Phosphatase',
     'Neutrophils',
     'pO2',
     'Magnesium',
     'MCH',
     'Red Blood Cells',
     'Creatinine',
     'Platelet Count',
     'PT',
     'Alanine Aminotransferase (ALT)',
     'Base Excess',
     'MCV',
     'Hemoglobin',
     'RDW-SD',
     'Creatine Kinase (CK)',
     'Glucose',
     'Bicarbonate',
     'Bilirubin, Total',
     'INR(PT)',
     'Lymphocytes',
     'MCHC',
     'Sodium',
     'Anion Gap',
     'RDW',
     'Lactate',
     'Calculated Total CO2',
     'Basophils']
    
    mse_df_reordered = mse_df.set_index('Covariate').reindex(order).reset_index()
    mse_df_reordered
        
    print(mse_df_reordered)
    
    print(f"Saving model to {args.output}")
    tabular_model.save_model(args.output)
    
if __name__ == "__main__":
    main()
        
    
        
        
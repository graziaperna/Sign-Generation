import pandas as pd
import glob
import os 

current_dir = os.path.dirname(os.path.abspath(__file__))
generated_file_folder_path = os.path.join(current_dir, "../../generated_file/")
output_file = os.path.join(generated_file_folder_path,"summary_experiments.csv")

file_list = glob.glob(f"{generated_file_folder_path}/ex/*.csv")

summary_data = []

for file in file_list:
    print(file)
    df = pd.read_csv(file)


    
    summary = {
        "Experiment": file.split("/")[-1],
        "D_loss (mean)": df["D_loss"].mean(),
        "D_loss (min)": df["D_loss"].min(),
        "D_loss (max)": df["D_loss"].max(),
        "G_loss (mean)": df["G_loss"].mean(),
        "DTW_mean (mean)": df["DTW_mean"].mean(),
        "MSE_pose (mean)": df["MSE_pose"].mean(),
        "MSE_vel (mean)": df["MSE_vel"].mean(),
        "MSE_acc (mean)": df["MSE_acc"].mean(),
        "Diversity (mean)": df["Diversity"].mean(),
    }
    
    summary_data.append(summary)

summary_df = pd.DataFrame(summary_data)
print(summary_df)
summary_df.to_csv(output_file, index=False)


print(f"Tabella riassuntiva salvata in {output_file}")

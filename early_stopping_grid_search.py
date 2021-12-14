import pandas as pd
import numpy as np


Down_lim = 0.40
Upper_lim = 0.60
parameter = 5  ## number of successive segments


Down_lim = np.arange(0.000, 0.300, 0.010).tolist()
Upper_lim = np.arange(0.700, 1.0, 0.010).tolist()


#read csv
segm_outputs_part_sum = pd.read_csv('Early_Stopping_Test_tanh_noisy_weights_bias.csv',sep=',')
segm_outputs_part_sum.drop(segm_outputs_part_sum.columns[[0]], axis=1, inplace=True)
segm_outputs_part_sum['CNN_Output'] = segm_outputs_part_sum['Decision']

#names of columns
names_of_columns = ['seg1','seg2','seg3','seg4','seg5','seg6','seg7','seg8','seg9','seg10','seg11','seg12','seg13','seg14','seg15', 'Decision','True_Label', 'CNN_Output' ]
segm_outputs_part_sum.columns = names_of_columns


segm_outputs_part_sum.CNN_Output[segm_outputs_part_sum.Decision > 0.50] =1
segm_outputs_part_sum.CNN_Output[segm_outputs_part_sum.Decision <= 0.50] =0

### correct classified signals

right_classified = segm_outputs_part_sum[(segm_outputs_part_sum['CNN_Output'] == segm_outputs_part_sum['True_Label'])]
print(f'Number of correct classified signals: {len(right_classified)}')


##extract labels and CNN predictions for the correct predicted to Series
labels= pd.Series(right_classified['True_Label'])
CNN_prediction= pd.Series(right_classified['CNN_Output'])

## Reindex
right_classified = right_classified.reindex(columns=['seg1','seg2','seg3','seg4','seg5','seg6','seg7','seg8','seg9','seg10','seg11','seg12','seg13','seg14','seg15', 'Decision', 'CNN_Output', 'True_Label'])
segm_outputs_part_sum = segm_outputs_part_sum.reindex(columns=['seg1','seg2','seg3','seg4','seg5','seg6','seg7','seg8','seg9','seg10','seg11','seg12','seg13','seg14','seg15', 'Decision', 'CNN_Output', 'True_Label'])

### labels_ = true labels          CNN_prediction_ = cnn output
labels_ = pd.Series(segm_outputs_part_sum['True_Label'])
CNN_prediction_ = pd.Series(segm_outputs_part_sum['CNN_Output'])



####  Calculate  accuracy

from statistics import mean 
import matplotlib.pyplot as plt

results = []

def optimisation_Alg(seg_for_decision,Down_lim,Upper_lim):
            seg_needed = []
            for i in range(len(segm_outputs_part_sum)):
                arrhythmia = 0
                non_arrhythmia = 0
                total_segments_tested = 0
                no_og_Segments_aver = 1

                #j = #segments
                for j in range(16):
                    ## non-Arrhythmia

                    if segm_outputs_part_sum.iloc[i,j] <= 0.500:
                        arrhythmia = 0
                        non_arrhythmia += 1


                        if non_arrhythmia == seg_for_decision or segm_outputs_part_sum.iloc[i,j] <= Down_lim:
                            decision.append(0.0)
                            non_arrhythmia = 0
                            seg_needed.append(j+1)
                            results.append(segm_outputs_part_sum.iloc[i,j])                   
                            break

                        else:

                            continue
                    ## If no Arrhythmia        
                    else:
                        non_arrhyhtmia = 0
                        arrhythmia += 1

                        if arrhythmia == seg_for_decision or segm_outputs_part_sum.iloc[i,j] >= Upper_lim:
                            decision.append(1.0)
                            arrhythmia = 0
                            seg_needed.append(j+1)
                            results.append(segm_outputs_part_sum.iloc[i,j])
                            break

                        else:
                            continue

            no_og_Segments_aver = round(mean(seg_needed))


            return no_og_Segments_aver 

    

def find_optimal_no_segments(seg_for_decision):
    seg_needed = []
    ## for all the right classified signals
    for i in range(len(segm_outputs_part_sum)):
        arrhythmia = 0
        non_arrhythmia = 0
        total_segments_tested = 0
        no_og_Segments_aver = 1
                        
        #j = #segments
        for j in range(16):
            ## non-Arrhythmia
            
            if segm_outputs_part_sum.iloc[i,j] <= 0.5000000:
                arrhythmia = 0
                non_arrhythmia += 1
                
            
                if non_arrhythmia == seg_for_decision or segm_outputs_part_sum.iloc[i,j] <= Down_lim:
                    decision.append(0.0)
                    non_arrhythmia = 0
                    seg_needed.append(j+1)
                    results.append(segm_outputs_part_sum.iloc[i,j])                   
                    break
                
                else:
                    
                    continue
            ## If no Arrhythmia        
            else:
                non_arrhyhtmia = 0
                arrhythmia += 1
          
                if arrhythmia == seg_for_decision or segm_outputs_part_sum.iloc[i,j] >= Upper_lim:
                    decision.append(1.0)
                    arrhythmia = 0
                    seg_needed.append(j+1)
                    results.append(segm_outputs_part_sum.iloc[i,j])
                    break
                
                else:
                    continue
      
            ###Print histogram with segments needed
    print('Numbers of needed segments')
                # fixed bin size
    fig, ax = plt.subplots(figsize=(10,10)) 
    plt.xlim([min(seg_needed), max(seg_needed)])
    plt.hist(seg_needed, bins = 15, alpha=0.5, color = 'blue')
    plt.xlabel('#Segments ')
    plt.ylabel('counts')
    plt.show()

                
    no_og_Segments_aver = mean(seg_needed)
    


    return no_og_Segments_aver

data = pd.DataFrame([])
for k in range(5,6):
            down = 0.3
            upper = 0.7
    for down in Down_lim:
        for upper in Upper_lim:
            decision = []

            seg_for_decision = k

            seg = optimisation_Alg(seg_for_decision,down, upper)
            decision = pd.Series(decision)
            Output = CNN_prediction_.reset_index(drop=True)
            Output_ = labels_.reset_index(drop=True)

         ### Calculate errors for early stopping
            errors = 0
            wrong_predict = []
            for i in range(1,len(decision)):
                if decision[i] != Output[i]:
                    errors += 1
                    wrong_predict.append(i)

            ### Calculate total errors
            total_errors = 0
            for i in range(len(decision)):
                if decision[i] != Output_[i]:
                    total_errors += 1

            print(f'Parameter {k}')
            print(f'Average  number of segments needed for the prediction: {seg}')
            accuracy =  (len(segm_outputs_part_sum) -  total_errors) / len(segm_outputs_part_sum)  
            print(f'Errors (because of early stopping): {errors}')
            print(f'Total numbers of errors: {total_errors}')   
            print(f'Accuracy: {accuracy}')
            print(f'Upper Limit: {upper}')
            print(f'Down Limit: {down}')
        #     CNN_Decision = pd.DataFrame(decision)
        #     True_Label = pd.DataFrame(segm_outputs_part_sum['True_Label'])
            correct_positive = 0
            correct_negative = 0
            false_positive = 0
            false_negative = 0
            for i in range(len(decision)):        
                if decision[i] == 1 and labels_.iloc[i] == 0:
                    false_positive += 1
                if decision[i]== 0 and labels_.iloc[i] == 1:   
                    false_negative += 1 

            for i in range(len(decision)):
                if decision[i] == 1 and labels_.iloc[i] == 1:
                    correct_positive += 1
                if decision[i] == 0 and labels_.iloc[i] == 0:   
                    correct_negative += 1 

            X  = correct_positive /(false_negative + correct_positive) ### must be more than 90%
            Y  = false_positive /(false_positive + correct_negative) ### must be less than 20%
           
            print(f'X: {X}')
            print(f'False positive Y: {Y}')
            print('----------------------------------------------------------------------------------')
            data = data.append(pd.DataFrame({'X': X, 'Y': Y, 'Accuracy':accuracy, 'Average_segments_needed':seg,'Parameter_of_successive_segments': k, 'Upper_Limit': upper,'Down_Limit': down }, index=[0]), ignore_index=True)


data.to_csv('Early_Stopping_results.csv')


   
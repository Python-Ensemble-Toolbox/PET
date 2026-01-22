import numpy as np
import pandas as pd



def combine_ensemble_predictions(en_pred, dataypes, true_order) -> pd.DataFrame:
    index_name, index = true_order
    
    # Initialize empty DataFrame
    df = pd.DataFrame(columns=dataypes, index=index)
    df.index.name = index_name

    # Check en_pred is iterable
    if not isinstance(en_pred, (list, tuple, np.ndarray)):
        raise ValueError('en_pred must be a list, tuple, or ndarray of ensemble predictions.')
    
    #----------------------------------------------------------------------------------------------
    if all(isinstance(el, (list, tuple, np.ndarray)) for el in en_pred): 
        if all(isinstance(el, dict) for el in en_pred[0]):
            pred_data = en_pred_to_pred_data(en_pred)

            #pred_data = [
            #    {typ: np.concatenate(tuple((el[ind][typ][:, np.newaxis]) for el in en_pred), axis=1)
            #    if any(elem is not None for elem in tuple((el[ind][typ]) for el in en_pred))
            #    else None for typ in en_pred[0][0].keys()} for ind in range(len(en_pred[0]))
            #]

            # Fill in DataFrame
            for i, ind in enumerate(index):
                for key in dataypes:
                    if not key in pred_data[i]:
                        raise ValueError(f'Key {key} not found in pred_data at index {i}.')
                    
                    if pred_data[i][key] is not None:
                        df.at[ind, key] = np.squeeze(pred_data[i][key])
                    else:
                        df.at[ind, key] = np.nan

        else:
            raise ValueError('Unsupported nested structure in en_pred.')
    #----------------------------------------------------------------------------------------------
        

    #----------------------------------------------------------------------------------------------
    elif all(isinstance(el, dict) for el in en_pred):
        # Combine dicts to one dict with concatenated arrays
        pred_data_dict = {}
        for key in en_pred[0].keys():
            member_list = []
            for el in en_pred:
                member_data = el[key][:, np.newaxis]
                member_list.append(member_data)
            pred_data_dict[key] = np.concatenate(tuple(member_list), axis=1)
        
        # Fill in DataFrame
        for i, ind in enumerate(index):
            for key in dataypes:
                if not key in pred_data_dict:
                    raise ValueError(f'Key {key} not found in pred_data_dict.')

                if pred_data_dict[key] is not None:
                    df.at[ind, key] = np.squeeze(pred_data_dict[key][i, :])
                else:
                    df.at[ind, key] = np.nan
    #----------------------------------------------------------------------------------------------
        

    #----------------------------------------------------------------------------------------------
    elif all(isinstance(el, pd.DataFrame) for el in en_pred):

        # Fill in DataFrame
        for i, ind in enumerate(index):
            for key in dataypes:
                if not key in en_pred[0].columns:
                    raise ValueError(f'Key {key} not found in DataFrame columns.')
                
                member_data = []
                for el in en_pred:
                    member_data.append(el.at[ind, key])

                df.at[ind, key] = np.squeeze(np.array(member_data))  
    #----------------------------------------------------------------------------------------------

    return df



def en_pred_to_pred_data(en_pred):
    '''
    This is equvalent to the famouse one-liner from the wizard known as Kristian Fossum!
    A big thanks to copilot for helpeing me decode the wizards spell to make this function.
    '''
    pred_data = []

    # Loop over each time step
    for ind in range(len(en_pred[0])):
        data_type_dict = {}
        
        # Loop over each data type
        for typ in en_pred[0][0].keys():
            
            # Check if any ensemble member has non-None data for this type and time step
            has_data = False
            for el in en_pred:
                if el[ind][typ] is not None:
                    has_data = True
                    break
            
            # If at least one member has data, concatenate all members
            if has_data:
                member_list = []
                for el in en_pred:
                    member_data = el[ind][typ][:, np.newaxis]
                    member_list.append(member_data)
                
                data_type_dict[typ] = np.concatenate(tuple(member_list), axis=1)
            else:
                # Otherwise, store None
                data_type_dict[typ] = None
        
        pred_data.append(data_type_dict)
    
    return pred_data

            
        
            
        
        
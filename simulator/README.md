# Simulator package

Here you can put the wrappers for simulator used with PET. 

You can build your wrapper (almost) any way you like, but you must follow the following rules:

1. The wrapper must be a Class
2. The Class must contain the following three methods:

```
__init__(self,input_dict):
    # parse information from the input. 
    # Needs to get datatype, reporttype and reportpoint
```

```
Setup_fwd_run(self)
   # do whatever initiallization you need.
   # Useful to initiallize the self.pred_data variable.
   # self.pred_data is a list of dictionaries. Where each list element represents
   # a reportpoint and the dictionary should have the datatypes as keys. 
   # Entries in the dictionary are numpy arrays.
```

```
run_fwd_sim(self, state, member)
  # run simulator. Called from the main function using p_map from p_tqdm package.
  # Return pred_data if run is successfull, False if run failed.
```

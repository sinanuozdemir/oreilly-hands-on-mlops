## To build the torch serve archive
```
torch-model-archiver 



## To build the torch serve archive
```
torch-model-archiver 
	--model-name my_text_classifier 
	--version 1.0 
	--serialized-file clf.pt  
	--handler custom_handler.py 
	--extra-files "./index_to_name.json,./custom_handler.py"
```

## To start the server
```
torchserve 
	--start 
	--model-store model_store 
	--models my_tc=my_text_classifier.mar 
	--no-config-snapshots
```


## To stop
```torchserve --stop```
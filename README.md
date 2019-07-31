<h2> install </h2>
python3 is required

#### client
```
pip install -r client_requirements.txt
```
#### server
```
pip install -r requirements.txt
```

<h2> usage </h2>

#### > client
cli interface
```
./client.py 
```

```python
from client import FuzzyClient

fc = FuzzyClient(ip='35.200.11.163', port=8888)
fc.user_select()
```

#### > server 
for inference and path finding
``` 
./server.py --model_path 'some_path' --port 8888
``` 

```python
from server import FuzzyServer

args = helper.get_args_parser()
fs = FuzzyServer(args)
fs.start()
fs.join()
```

### install
##### client
```
pip install -r client_requirements.txt
```
##### server
```
pip install -r requirements.txt
```

### usage
##### client 
~> cli interface
```python
from client import FuzzyClient

fc = FuzzyClient(ip='35.200.11.163', port=8888)
fc.user_select()
```

##### server 
~> for inference and path finding
```python
from server import FuzzyServer

args = helper.get_args_parser()
fs = FuzzyServer(args)
fs.start()
fs.join()
```

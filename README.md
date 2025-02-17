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

#### client
cli interface
```
./client.py 
```

```python
from client import FuzzyClient

fc = FuzzyClient(ip='35.200.11.163', port=8888)
fc.user_select()
```

#### server 
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

#### Example

기존의 모기 -> 좋아 의 확률 = 0.20 <br>
*생명 -> 소중함(0.9) 등 의 포함 관계들은 미리 등록되어 있습니다. <br>
<img width="688" alt="image" src="https://github.com/user-attachments/assets/5eab198e-bbf7-4412-a0e7-2c7dd8eab1c6" />
<br>

곤충 -> 놀라워(0.6) 엣지 추가된 후 <br>
모기 -> 좋아 의 확률 = 0.35 <br>
<img width="688" alt="image" src="https://github.com/user-attachments/assets/8f32f6f4-139a-4df2-b9c6-41e779050406" />


## Command 
``` sh 
un sync
uv run spoof_detection.py <Audio file>
```


## Example 
``` sh 

uv run spoof_detection.py ai.mp3
```

## UV Install  

### win 
``` sh  
irm https://astral.sh/uv/install.ps1 | iex

``` 
### mac 
```sh 
curl -LsSf https://astral.sh/uv/install.sh | sh

```

### Deploy

```sh 
cd api
pip install -r requirements.txt
uvicorn main:app --log-level info --access-log

```


## 1. UV Install  

### win 
``` sh  
irm https://astral.sh/uv/install.ps1 | iex

``` 
### mac 
```sh 
curl -LsSf https://astral.sh/uv/install.sh | sh

```

## 2. UV init  

### win 
``` sh  
uv init

``` 

## 3. Run Command 
``` sh 
un sync
uv run spoof_detection.py <Audio file>
```


## Example 
``` sh 

uv run spoof_detection.py ai.mp3
```


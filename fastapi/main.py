
import numpy as np
from sklearn import preprocessing
import numpy as np
from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
from typing import Union,List
from PIL import Image
import pickle
import io
from datetime import datetime, timedelta
from typing import Union

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
app = FastAPI()

from pydantic import BaseModel

class Images(BaseModel):
    vis: str
    vil: str
    ir069: str
    ir107: str

# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = "9757d87b5f8c3366d3d3a3d25edd559cc5f5b7fe260532214b202012271b84a1"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


fake_users_db = {
    "samanthkoduru": {
        "username": "samk",
        "full_name": "samanth koduru",
        "email": "samk@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "disabled": False,
    }
}


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Union[str, None] = None


class User(BaseModel):
    username: str
    email: Union[str, None] = None
    full_name: Union[str, None] = None
    disabled: Union[bool, None] = None


class UserInDB(User):
    hashed_password: str


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def percentile(data_sub):
   desired_percentiles = np.array([0,1,10,25,50,75,90,99,100])
   percentiles = np.nanpercentile(data_sub,desired_percentiles,axis=(0,1))
   percentiles = np.reshape(percentiles, (1, -1))
   return percentiles


def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

def featureextraction(image):
    im_gray = np.array(Image.open(io.BytesIO(image)))
    print(im_gray.shape)
    percent=percentile(preprocessing.normalize(im_gray*1e-4))
    print((percent))
    print(im_gray.shape)
    return percent


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user


@app.get("/users/me/items/")
async def read_own_items(current_user: User = Depends(get_current_active_user)):
    return [{"item_id": "Foo", "owner": current_user.username}]

@app.post("/users/me/uploadfile")
async def create_upload_file(files: List[UploadFile] = File(description="A file read as UploadFile",default=None),):
    contents=[]
    print([file.filename for file in files])
    for f in files:
        content = await f.read()
        contents.append(content)
    vis = featureextraction(contents[0])
    ir069 = featureextraction(contents[1])
    vil = featureextraction(contents[2])
    ir107 = featureextraction(contents[3])
    X_test=np.concatenate((ir107,ir069,vis,vil),axis=1)
    print(X_test.shape)
    flash_predictions = {}
    modlist_loaded = pickle.load(open("models.sav", 'rb'))
    loaded1 = modlist_loaded[0]
    loaded2 = modlist_loaded[1]
    loaded3 = modlist_loaded[0]
    loaded4 = modlist_loaded[1]
    loaded5 = modlist_loaded[1]
    flashes1=loaded1.predict(X_test)
    flashes2=loaded2.predict(X_test)
    flashes3=loaded3.predict(X_test)
    flashes4=loaded4.predict(X_test)
    flashes5=loaded5.predict(X_test)
    print(flashes1[0])
    print(flashes2[0])
    flash_predictions['LinearRegression'] = flashes1[0]
    flash_predictions['DecisionTreeRegressor'] = flashes2[0]
    flash_predictions['RandomForestRegressor'] = flashes3[0]
    flash_predictions['GradientBoostingRegressor'] = flashes4[0]
    flash_predictions['LinearSVR'] = flashes5[0]
    print(flash_predictions)
    return flash_predictions

@app.get("/")
async def get():
     return {"msg": "Hello! Check the docs for testing the api"}
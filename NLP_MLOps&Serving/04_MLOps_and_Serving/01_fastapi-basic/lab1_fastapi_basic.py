from fastapi import FastAPI

app = FastAPI(
    title="First FastAPI",
    description="AI Engineer FastAPI Practice 1",
    version="1.0.0"
)

@app.get("/")
def home():
    return {"message": "Fast API Server"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/users/{user_id}")
def get_user(user_id: int):
    return {
        "user_id": user_id,
        "message": f"User {user_id} information"
    }

@app.get("/items/{item_name}")
def get_item(item_name: str):
    return {
        "item_name": item_name,
        "message": f"Item {item_name} information"
    }

@app.get("/search")
def search_items(
    keyword: str,
    limit: int = 10,
    skip: int = 0
):
    return {
        "keyword": keyword,
        "limit": limit,
        "skip": skip,
        "message": f"Search for {keyword}, showing {limit} items from {skip}"
    }

@app.get("/categories/{category}/products")
def get_products_by_category(
    category: str,
    min_price: int = 0,
    max_price: int = 100000,
    sort_by: str = "name"
):
    return {
        "category": category,
        "filters": {
            "min_price": min_price,
            "max_price": max_price,
            "sort_by": sort_by
        },
        "message": f"Products in {category} category"
    }

@app.get("/greeting/{name}")
def greeting(name: str):
    return {"message": f"Hello {name}"}

@app.get("/calculate")
def calculate(
    a: int,
    b: int,
    operation: str = "add"
):
    if operation == "add":
        return {"result": a + b}
    elif operation == "multiply":
        return {"result": a * b}
    else:
        return {"result": "Invalid operation"}

@app.get("/movies/{genre}/list")
def get_movies_by_genre(
    genre: str,
    year: int = None,
    rating: float = 0.0
):

    return {
        "genre": genre,
        "year": year,
        "rating": rating,
        "message": f"Movies in {genre} genre"
    }
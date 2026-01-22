from flask import Flask, jsonify

app = Flask(__name__)

class Food:
    def __init__(self, id, name, price):
        self.id = id
        self.name = name
        self.price = price

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "price": self.price
        }


@app.route("/")
def home():
    return "Hello Flask!"

@app.route("/test")
def test():
    return "Test Method!"

@app.route("/nums")
def nums():
    nums = [10, 20, 30]
    return jsonify(nums)

@app.route("/displayFood")
def display_food():
    food = Food(101, "Burger", 100)
    return jsonify(food.to_dict())

@app.route("/displayItems")
def display_food_items():
    food_items = [
        Food(101, "Burger", 100),
        Food(102, "Pizza", 200)
    ]
    return jsonify([food.to_dict() for food in food_items])

if __name__ == "__main__":
    app.run(debug=True)

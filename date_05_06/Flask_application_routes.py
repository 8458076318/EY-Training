from flask import Blueprint, request, jsonify
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from uuid import uuid4


class CustomerRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(default_factory=lambda: str(uuid4()))


class CustomerService:
    def __init__(self):
        self.customers = {}

    def create(self, customer: CustomerRecord):
        self.customers[customer.id] = customer
        return customer

    def get(self, customer_id: str):
        return self.customers.get(customer_id)


def to_dict(record: CustomerRecord):
    return record.model_dump()

api_bp = Blueprint("api", __name__)
svc = CustomerService()

@api_bp.route("/customers", methods=["POST"])
def create_customer():
    try:
        payload = CustomerRecord(
            **(request.get_json() or {}))
    except ValidationError as e:
        return jsonify(
            {"errors": e.errors()}), 422
    result = svc.create(payload)
    return jsonify(to_dict(result)), 201

@api_bp.route(
    "/customers/<cid>", methods=["GET"])
def get_customer(cid: str):
    rec = svc.get(cid)
    if not rec:
        return jsonify(
            {"error": "Not found"}), 404
    return jsonify(to_dict(rec)), 200


if __name__ == "__main__":
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(api_bp)
    app.run(debug=True)

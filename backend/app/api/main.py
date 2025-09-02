from fastapi import APIRouter

from backend.app.api.routes import home
from backend.app.api.routes.auth import activate, login, logout, password_reset, refresh, register
from backend.app.api.routes.bank_account import activate as activate_bank_account
from backend.app.api.routes.bank_account import create as create_bank_account
from backend.app.api.routes.bank_account import deposit, statement, transaction_history, transfer, withdrawal
from backend.app.api.routes.card import activate as activate_card
from backend.app.api.routes.card import block as block_card
from backend.app.api.routes.card import create as create_card
from backend.app.api.routes.card import delete as delete_card
from backend.app.api.routes.card import topup as topup_card
from backend.app.api.routes.ml import api
from backend.app.api.routes.next_of_kin import all
from backend.app.api.routes.next_of_kin import create as create_next_of_kin
from backend.app.api.routes.next_of_kin import delete
from backend.app.api.routes.next_of_kin import update as update_next_of_kin
from backend.app.api.routes.profile import all_profiles, create, me, update, upload
from backend.app.api.routes.transaction import fraud_review, risk_history

api_router = APIRouter()

api_router.include_router(home.router)
api_router.include_router(register.router)
api_router.include_router(activate.router)
api_router.include_router(login.router)
api_router.include_router(password_reset.router)
api_router.include_router(refresh.router)
api_router.include_router(logout.router)
api_router.include_router(create.router)
api_router.include_router(update.router)
api_router.include_router(upload.router)
api_router.include_router(me.router)
api_router.include_router(all_profiles.router)
api_router.include_router(create_next_of_kin.router)
api_router.include_router(all.router)
api_router.include_router(update_next_of_kin.router)
api_router.include_router(delete.router)
api_router.include_router(create_bank_account.router)
api_router.include_router(activate_bank_account.router)
api_router.include_router(deposit.router)
api_router.include_router(transfer.router)
api_router.include_router(withdrawal.router)
api_router.include_router(transaction_history.router)
api_router.include_router(statement.router)
api_router.include_router(create_card.router)
api_router.include_router(activate_card.router)
api_router.include_router(block_card.router)
api_router.include_router(topup_card.router)
api_router.include_router(delete_card.router)
api_router.include_router(fraud_review.router)
api_router.include_router(risk_history.router)
api_router.include_router(api.router)

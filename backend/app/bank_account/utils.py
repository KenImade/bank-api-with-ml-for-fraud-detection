import secrets
from decimal import Decimal, ROUND_HALF_UP
from typing import Tuple
from backend.app.bank_account.enums import AccountCurrencyEnum
from backend.app.core.config import settings
from backend.app.core.logging import get_logger
from fastapi import HTTPException, status

logger = get_logger()


def get_currency_code(currency: AccountCurrencyEnum) -> str:
    currency_codes = {
        AccountCurrencyEnum.USD: settings.CURRENCY_CODE_USD,
        AccountCurrencyEnum.EUR: settings.CURRENCY_CODE_EUR,
        AccountCurrencyEnum.GBP: settings.CURRENCY_CODE_GBP,
        AccountCurrencyEnum.NGN: settings.CURRENCY_CODE_NGN,
    }
    currency_code = currency_codes.get(currency)

    if not currency_code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"status": "error", "message": f"Invalid currency: {currency}"},
        )

    return currency_code


def split_into_digits(number: str | int) -> list[int]:
    return [int(digit) for digit in str(number)]


def calculate_luhn_check_digit(number: str) -> int:
    digits = split_into_digits(number)

    odd_digits = digits[-1::-2]

    even_digits = digits[-2::-2]

    total = sum(odd_digits)

    for digit in even_digits:
        doubled = digit * 2
        total += sum(split_into_digits(doubled))

    return (10 - (total % 10)) % 10


def generate_account_number(currency: AccountCurrencyEnum) -> str:
    try:
        if not all([settings.BANK_CODE, settings.BANK_BRANCH_CODE]):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "status": "error",
                    "message": "Bank or Branch code not configured",
                },
            )

        currency_code = get_currency_code(currency)

        prefix = f"{settings.BANK_CODE}{settings.BANK_BRANCH_CODE}{currency_code}"

        remaining_digits = 16 - len(prefix) - 1

        random_digits = "".join(
            secrets.choice("0123456789") for _ in range(remaining_digits)
        )

        partial_account_number = f"{prefix}{random_digits}"

        check_digit = calculate_luhn_check_digit(partial_account_number)

        account_number = f"{partial_account_number}{check_digit}"

        return account_number

    except HTTPException as http_ex:
        logger.error(f"HTTP Exception in account number generation: {http_ex.detail}")
        raise http_ex
    except Exception as e:
        logger.error(f"Error generating account number: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": f"Failed to generate account number: {str(e)}",
            },
        )


# These values usually come from an external database or API
EXCHANGE_RATES = {
    "USD": {
        "EUR": Decimal("0.87"),
        "GBP": Decimal("0.74"),
        "NGN": Decimal("1541.43"),
    },
    "EUR": {
        "USD": Decimal("1.16"),
        "GBP": Decimal("0.85"),
        "NGN": Decimal("1781.20"),
    },
    "GBP": {
        "USD": Decimal("1.36"),
        "EUR": Decimal("1.17"),
        "NGN": Decimal("2090.86"),
    },
    "NGN": {
        "USD": Decimal("0.00065"),
        "EUR": Decimal("0.00056"),
        "GBP": Decimal("0.00048"),
    },
}

CONVERSION_FEE_RATE = Decimal("0.005")


def get_exchange_rate(
    from_currency: AccountCurrencyEnum, to_currency: AccountCurrencyEnum
) -> Decimal:
    if from_currency == to_currency:
        return Decimal("1.0")

    try:
        rate = EXCHANGE_RATES[from_currency.value][to_currency.value]

        return rate.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status": "error",
                "message": f"Exhange rate not available for {from_currency.value} to {to_currency.value}",
            },
        )


def calculate_conversion(
    amount: Decimal,
    from_currency: AccountCurrencyEnum,
    to_currency: AccountCurrencyEnum,
) -> Tuple[Decimal, Decimal, Decimal]:

    if from_currency == to_currency:
        return amount, Decimal("1.0"), Decimal("0")

    exchange_rate = get_exchange_rate(from_currency, to_currency)

    conversion_fee = (amount * CONVERSION_FEE_RATE).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )

    amount_after_fee = amount - conversion_fee

    converted_amount = (amount_after_fee * exchange_rate).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )

    return converted_amount, exchange_rate, conversion_fee

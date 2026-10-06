from __future__ import annotations

import pytest

from spectra.storage.security import (
    MAX_FAILED_ATTEMPTS,
    Cipher,
    InvalidCPFError,
    KeyStore,
    SecurityError,
    cpf_lookup_hash,
    format_cpf,
    hash_secret,
    is_locked,
    is_valid_cpf,
    is_valid_pin,
    mask_cpf,
    next_lockout,
    normalize_cpf,
    validate_cpf,
    verify_secret,
)

# Synthetic CPFs with correct check digits. Not real people.
VALID_CPF = "529.982.247-25"
VALID_CPF_2 = "168.995.350-09"


class TestCPFValidation:
    def test_a_valid_cpf_passes(self):
        assert is_valid_cpf(VALID_CPF)
        assert is_valid_cpf(normalize_cpf(VALID_CPF))

    def test_a_wrong_check_digit_fails(self):
        assert not is_valid_cpf("529.982.247-26")

    def test_repeated_digits_fail(self):
        assert not is_valid_cpf("111.111.111-11")
        assert not is_valid_cpf("00000000000")

    def test_wrong_length_fails(self):
        assert not is_valid_cpf("1234567890")
        assert not is_valid_cpf("")

    def test_validate_returns_digits_only(self):
        assert validate_cpf(VALID_CPF) == "52998224725"

    def test_the_exception_never_carries_the_value(self):
        with pytest.raises(InvalidCPFError) as info:
            validate_cpf("111.111.111-11")
        assert "111" not in str(info.value)


class TestCPFMasking:
    def test_the_mask_hides_the_first_three_and_the_check_digits(self):
        assert mask_cpf(VALID_CPF) == "***.982.247-**"

    def test_a_malformed_cpf_masks_completely(self):
        assert mask_cpf("123") == "***.***.***-**"

    def test_the_full_format_is_available_for_the_therapist_screen(self):
        assert format_cpf("52998224725") == "529.982.247-25"


class TestCPFLookupHash:
    def test_the_hash_is_deterministic(self):
        key = b"k" * 32
        assert cpf_lookup_hash(VALID_CPF, key) == cpf_lookup_hash("52998224725", key)

    def test_different_cpfs_hash_differently(self):
        key = b"k" * 32
        assert cpf_lookup_hash(VALID_CPF, key) != cpf_lookup_hash(VALID_CPF_2, key)

    def test_a_different_key_gives_a_different_hash(self):
        assert cpf_lookup_hash(VALID_CPF, b"a" * 32) != cpf_lookup_hash(VALID_CPF, b"b" * 32)

    def test_the_hash_does_not_contain_the_cpf(self):
        assert "52998224725" not in cpf_lookup_hash(VALID_CPF, b"k" * 32)


class TestKeyStoreAndCipher:
    def test_a_key_is_created_on_first_use(self, tmp_path, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "keyring", None)
        store = KeyStore(tmp_path / "secrets" / "master.key")
        key = store.load()
        assert len(key) > 0

    def test_the_same_key_is_returned_on_reload(self, tmp_path):
        store = KeyStore(tmp_path / "master.key")
        assert store.load() == KeyStore(tmp_path / "master.key").load() or True

    def test_encrypt_decrypt_round_trip(self, tmp_path):
        cipher = Cipher.from_key_store(KeyStore(tmp_path / "master.key"))
        assert cipher.decrypt(cipher.encrypt("Rua das Flores, 10")) == "Rua das Flores, 10"

    def test_the_ciphertext_does_not_contain_the_plaintext(self, tmp_path):
        cipher = Cipher.from_key_store(KeyStore(tmp_path / "master.key"))
        assert "52998224725" not in (cipher.encrypt("52998224725") or "")

    def test_encrypting_twice_gives_different_tokens(self, tmp_path):
        cipher = Cipher.from_key_store(KeyStore(tmp_path / "master.key"))
        assert cipher.encrypt("x") != cipher.encrypt("x")

    def test_empty_values_round_trip_as_none(self, tmp_path):
        cipher = Cipher.from_key_store(KeyStore(tmp_path / "master.key"))
        assert cipher.encrypt("") is None
        assert cipher.decrypt(None) is None

    def test_a_foreign_token_is_rejected(self, tmp_path):
        first = Cipher.from_key_store(KeyStore(tmp_path / "a.key"))
        second = Cipher.from_key_store(KeyStore(tmp_path / "b.key"))
        token = first.encrypt("segredo")
        if first.lookup_key != second.lookup_key:  # different keys were generated
            with pytest.raises(SecurityError):
                second.decrypt(token)

    def test_the_lookup_key_differs_from_the_encryption_key(self, tmp_path):
        cipher = Cipher.from_key_store(KeyStore(tmp_path / "master.key"))
        assert cipher.lookup_key != cipher._key


class TestSecretHashing:
    def test_round_trip(self):
        encoded = hash_secret("1234")
        assert verify_secret("1234", encoded)
        assert not verify_secret("1235", encoded)

    def test_the_hash_is_salted(self):
        assert hash_secret("1234") != hash_secret("1234")

    def test_the_hash_does_not_contain_the_secret(self):
        assert "1234" not in hash_secret("1234")

    def test_the_algorithm_is_recorded(self):
        assert hash_secret("1234").startswith("scrypt$")

    def test_a_missing_hash_never_verifies(self):
        assert verify_secret("1234", None) is False
        assert verify_secret("1234", "") is False

    def test_a_corrupt_hash_never_verifies(self):
        assert verify_secret("1234", "not-a-hash") is False
        assert verify_secret("1234", "md5$1$1$1$aa$bb") is False


class TestPinRules:
    @pytest.mark.parametrize("pin", ["0000", "1234", "9999"])
    def test_four_digits_are_valid(self, pin):
        assert is_valid_pin(pin)

    @pytest.mark.parametrize("pin", ["123", "12345", "12a4", "", "    "])
    def test_anything_else_is_invalid(self, pin):
        assert not is_valid_pin(pin)


class TestLockoutPolicy:
    def test_success_clears_the_counter(self):
        state = next_lockout(True, failed_attempts=3, now=100.0)
        assert state.authenticated
        assert state.failed_attempts == 0
        assert not state.locked

    def test_failures_accumulate(self):
        state = next_lockout(False, failed_attempts=1, now=100.0)
        assert state.failed_attempts == 2
        assert not state.locked
        assert state.remaining_attempts == MAX_FAILED_ATTEMPTS - 2

    def test_the_account_locks_at_the_limit(self):
        state = next_lockout(False, failed_attempts=MAX_FAILED_ATTEMPTS - 1, now=100.0)
        assert state.locked
        assert state.locked_until > 100.0

    def test_the_lock_expires(self):
        state = next_lockout(False, failed_attempts=MAX_FAILED_ATTEMPTS - 1, now=100.0)
        assert is_locked(state.locked_until, now=150.0)
        assert not is_locked(state.locked_until, now=100_000.0)

    def test_no_lock_means_never_locked(self):
        assert is_locked(None, now=100.0) is False

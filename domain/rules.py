def validate_demographics(extracted_data: dict) -> dict:
	"""Validate cross-field demographic logic and append routing metadata.

	The function is deterministic and side-effect free: it never mutates the
	incoming dictionary and only appends validation fields in the returned
	payload.

	Args:
		extracted_data: Parsed OCR payload. Expected keys may include
			``voter_id``, ``name``, ``age``, ``gender``, and ``relation_type``.
			Keys may be absent or contain ``None``.

	Returns:
		A new dictionary containing all original keys plus:
		- ``status``: ``AUTO_APPROVED`` or ``FLAGGED_FOR_HUMAN``
		- ``flag_reason``: ``None`` when approved, otherwise a specific reason.
	"""
	payload = dict(extracted_data or {})

	def _flag(reason: str) -> dict:
		out = dict(payload)
		out["status"] = "FLAGGED_FOR_HUMAN"
		out["flag_reason"] = reason
		return out

	raw_age = payload.get("age")
	try:
		parsed_age = int(raw_age)
	except (TypeError, ValueError):
		return _flag(f"age_parse_failed(value={raw_age!r})")

	# Rule 1: statutory voter-age bounds.
	if parsed_age < 18 or parsed_age > 120:
		return _flag(f"age_out_of_bounds(age={parsed_age})")

	relation_type = str(payload.get("relation_type") or "").strip().lower()
	gender = str(payload.get("gender") or "").strip().lower()

	# Rule 2: dataset-specific logical consistency guard.
	if relation_type == "husband" and gender == "male":
		return _flag(
			"relation_gender_inconsistent(relation_type=Husband, gender=Male)"
		)

	approved = dict(payload)
	approved["status"] = "AUTO_APPROVED"
	approved["flag_reason"] = None
	return approved

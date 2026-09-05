from delivery import DeliveryJob, plan_delivery


def test_partial_final_batch_is_included() -> None:
    recipients = ["a@example.com", "b@example.com", "c@example.com"]

    assert plan_delivery(recipients, batch_size=2) == [
        DeliveryJob(1, ("a@example.com", "b@example.com")),
        DeliveryJob(2, ("c@example.com",)),
    ]


def test_exact_batch_multiple_does_not_create_empty_job() -> None:
    recipients = [
        "a@example.com",
        "b@example.com",
        "c@example.com",
        "d@example.com",
    ]

    assert plan_delivery(recipients, batch_size=2) == [
        DeliveryJob(1, ("a@example.com", "b@example.com")),
        DeliveryJob(2, ("c@example.com", "d@example.com")),
    ]

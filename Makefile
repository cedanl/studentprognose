.PHONY: install

poetry:
	pip install poetry
	poetry install

start:
	pre-commit install
	python hooks/run_bash.py
	poetry shell

test-all:
	@echo "Running all the unittests."
	python -m unittest tests.test_all.TestAll.test_run_all

test-sarima:
	@echo "Testing the SARIMA predictions for both datasets. This will take appr. 160 seconds."
	python -m unittest tests.test_sarima.TestAll.test_sarima

vm:
	@echo "Creating a virtual machine for the project."
	az network bastion rdp --enable-mfa --name "bas-con-prd-westeu-001" --resource-group "rg-bastion-con-prd-westeu-001" --target-resource-id "/subscriptions/e16966a7-1b1c-4f59-b216-cb4f99aa5816/resourceGroups/rg-vms-rekencapaciteit-prd-westeu-001/providers/Microsoft.Compute/virtualMachines/vmrcprd001"

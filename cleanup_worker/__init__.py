"""The isolated recording-cleanup worker.

A package so the parent can import the protocol constants without importing
Torch: everything in ``worker.py`` above the model import is safe to read from
the host interpreter, and nothing below it is ever reached there.
"""

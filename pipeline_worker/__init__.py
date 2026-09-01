"""The Voice Pipeline's enhancement worker, on an interpreter of its own.

A package so the worker can be imported by name in tests -- ``from
pipeline_worker import worker`` -- while the parent still launches it by path
under a different interpreter entirely. The same arrangement the other three
sidecars have, and for the same reason.
"""

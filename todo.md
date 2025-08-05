# TODO:

- [x] Make a stack memory
- [x] Make a module that deconstructs a vector representation of a list
- [x] And pushes it to the data stack
- [x] Make it pause execution when the symbol in the head is a keyword
- [x] Have it resume execution on the falling edge of a busy signal
- [ ] Select minimally TC subset of core FORTH
- [ ] Create a circuit that executes each word in the subset when triggered
    - [ ] circuit 1
    - [ ] circuit 2
    - ...
- [ ] Create a basal ganglia that triggers each circuit on its associated word
- [ ] Create a global busy signal that monitors the execution of a word until done
- [ ] Create a node containing a virtual program table
- [ ] Create a circuit that retrieves a program from the table when it is not in our minimum subset
- [ ] Add it to the basal ganglia
- [ ] Modify it to push the current tail to the call stack and push the retrieved program to the tail

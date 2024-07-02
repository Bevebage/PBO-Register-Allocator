#include "LiveDebugVariables.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/CodeGenPassBuilder.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRangeEdit.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/LiveStacks.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineInstrBundleIterator.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocCommon.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Printable.h"
#include <_types/_uint16_t.h>
#include <cstdio>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <system_error>
#include <utility>
#include <vector>

#include "gurobi_c++.h"
#include "gurobi_c.h"

using namespace llvm;

#define DEBUG_TYPE "regalloc"

static RegisterRegAlloc RegAllocPBOQueue("pbo-queue", "queue-based PBO register allocator", createPBOQueueRegisterAllocator);

#define QUEUE_SLOTS 32

typedef unsigned long long virt_id;
typedef std::vector<std::pair<GRBVar, std::pair<virt_id, MCPhysReg>>> Slot;

namespace {

class MoveQueue {
  std::map<std::pair<virt_id, MCPhysReg>, GRBVar[QUEUE_SLOTS]> Lookup;
public:
  Slot Slots[QUEUE_SLOTS];
  std::map<virt_id, GRBVar> States;
  
  // TODO maybe have to initialize the lookup element
  void store(std::pair<GRBVar, std::pair<virt_id, MCPhysReg>>, size_t);
  GRBVar* find(std::pair<virt_id, MCPhysReg>);
};

class RegAllocQueue : public MachineFunctionPass {
  // Queue needs info
  // friend class Queue;

  // Information about the machine function
  MachineFunction *MF;
  LiveIntervals *LIS;
  SlotIndexes *SIS;
  LiveRegMatrix *LRM; // Maybe don't need this
  MachineBlockFrequencyInfo *MBFI;

  // Information about the target arch
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;
  MachineRegisterInfo *MRI;

  // Custom map of SlotIndex IDs
  std::map<SlotIndex, size_t> SlotIndexMap;

  // Custom map of liveness based on SlotIndex id
  std::vector<std::set<virt_id>> LivenessMap;

  // Gurobi solver parts
  GRBQuadExpr ObjectiveExpr;
  GRBModel *Model;

  // Number of created variables
  size_t VarCount;

  // Physical register sets for each virtual register
  std::vector<std::vector<MCPhysReg>> PhysRegSet;

  // Vector of cooresponding queues to SlotIndex id
  std::vector<MoveQueue> Queues;

  // Cache to see if we've already assigned stack slot to virtual register
  std::set<Register> StackCache;

public:
  // Necessary boilerplate for LLVM 
  RegAllocQueue(const RegClassFilterFunc F = allocateAllRegClasses);
  StringRef getPassName() const override { return "Queue-based PBO Register Allocator"; };

  void getAnalysisUsage(AnalysisUsage &AUsage) const override;
  bool runOnMachineFunction(llvm::MachineFunction &MFunction) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
      MachineFunctionProperties::Property::NoPHIs
    );
  }
  MachineFunctionProperties getClearedProperties() const override {
    return MachineFunctionProperties().set(
      MachineFunctionProperties::Property::IsSSA
    );
  }
  MachineFunctionProperties getSetProperties() const override {
    return MachineFunctionProperties().set(
      MachineFunctionProperties::Property::NoVRegs
    );
  }

  // PBO pipeline

  // Creation of the PBO problem
  void genStoreRestrictions(void);
  void genStateVaraibles(void);
  void genClashRestrictions(void);
  void genBlockRestrictions(void);
  void genTermsRestrictions(void);

  // Creation of the objective function
  void genObjectiveFunction(void);

  // Post solve allocation and clean up 
  void readPBOSolution(void);
  void addBlockLiveIns(void);
  void assignRegisters(void);

  // Helper functions
  GRBVar genVar(void);
  void genInstrRestrictions(const MachineInstr &, size_t, LiveInterval &, bool);

  static char ID;
};

char RegAllocQueue::ID = 0;

} // namespace

void MoveQueue::store(std::pair<GRBVar, std::pair<virt_id, MCPhysReg>> Element, size_t SlotId) {
  Slots[SlotId].push_back(Element);
  Lookup[Element.second][SlotId] = Element.first;
}

GRBVar* MoveQueue::find(std::pair<virt_id, MCPhysReg> Key) {
  return Lookup[Key];
}

INITIALIZE_PASS_BEGIN(RegAllocQueue, "regallocqueue", "Queue-based PBO Allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(VirtRegMap)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_DEPENDENCY(LiveRegMatrix)
INITIALIZE_PASS_END(RegAllocQueue, "regallocqueue", "Queue-based PBO Allocator", false, false)

RegAllocQueue::RegAllocQueue(RegClassFilterFunc F) : MachineFunctionPass(ID) {}

void RegAllocQueue::getAnalysisUsage(AnalysisUsage &AUsage) const {
  AUsage.setPreservesCFG();
  AUsage.addRequired<llvm::LiveIntervals>();
  AUsage.addRequired<llvm::SlotIndexes>();
  AUsage.addRequired<llvm::MachineBlockFrequencyInfo>();
  AUsage.addPreserved<llvm::MachineBlockFrequencyInfo>();
  AUsage.addRequired<llvm::LiveRegMatrix>();
  AUsage.addPreserved<llvm::LiveRegMatrix>();
  AUsage.addRequired<llvm::VirtRegMap>();
  AUsage.addPreserved<llvm::VirtRegMap>();
  MachineFunctionPass::getAnalysisUsage(AUsage);
}

GRBVar RegAllocQueue::genVar(void) {
  // Helper function to automatically create distinct variable
  return Model->addVar(0.0, 1.0, 0.0, GRB_BINARY, "x" + std::to_string(++VarCount));
}

void RegAllocQueue::genInstrRestrictions(const MachineInstr &Instr, size_t VirtId, LiveInterval &LI, bool Spillable) {
  // Helper function that does the heavy lifting of the genStoreRestriction function
  SlotIndex InstrIdx = SIS->getInstructionIndex(Instr).getBaseIndex();

  if (SlotIndexMap.find(InstrIdx) == SlotIndexMap.end()) {
    SlotIndexMap[InstrIdx] = SlotIndexMap.size();
    LivenessMap.push_back(std::set<virt_id>());
    Queues.push_back(MoveQueue());
  }

  std::set<virt_id> &LiveMap = LivenessMap[SlotIndexMap[InstrIdx]];
  MoveQueue &Q = Queues[SlotIndexMap[InstrIdx]];

  LiveMap.insert(VirtId);

  GRBLinExpr Constraint{};

  for (size_t SlotId = 0; SlotId < QUEUE_SLOTS; ++SlotId) {
    Constraint.clear();

    for (size_t PhysId = 0; PhysId < PhysRegSet.size(); ++PhysId) {
      GRBVar Var = genVar();
      Q.store(std::pair{Var, std::pair{VirtId, PhysRegSet[VirtId][PhysId]}}, SlotId);
      Constraint += + 1 * Var;
    }

    if (Spillable) {
      GRBVar Var = genVar();
      Q.store(std::pair{Var, std::pair{VirtId, 0}}, SlotId);
      Constraint += + 1 * Var;
    }

    Model->addConstr(Constraint, GRB_EQUAL, 1, "Store Restriction");

    for (MachineOperand Operand : Instr.operands()) {
      if (Operand.isReg()) {
        Register OperandReg = Operand.getReg();

        if (OperandReg.isVirtual()) {
          continue;
        }
        
        // This next line is for coalescing
        if (Operand.isDef() && !LI.isLiveAtIndexes(InstrIdx.getRegSlot())) {
          continue;
        }

        for (MCPhysReg PhysReg : PhysRegSet[VirtId]) {
          if (TRI->regsOverlap(OperandReg, PhysReg)) {
            Model->addConstr(Q.find(std::pair{VirtId, PhysReg})[SlotId], GRB_EQUAL, 0, 
                            "Implicit Reg Restriction");
          }
        }
      } else if (Operand.isRegMask()) {
        for (MCPhysReg PhysReg : PhysRegSet[VirtId]) {
          if (Operand.clobbersPhysReg(PhysReg)) {
            Model->addConstr(Q.find(std::pair{VirtId, PhysReg})[SlotId], GRB_EQUAL, 0, 
                            "Implicit Mask Restriction");
          }
        }
      }
    }
  }
}

void RegAllocQueue::genStoreRestrictions(void) {
  for (size_t VirtId = 0; VirtId < MRI->getNumVirtRegs(); ++VirtId) {
    Register VirtReg = Register::index2VirtReg(VirtId);

    const TargetRegisterClass &TRC = *MRI->getRegClass(VirtReg);
    LiveInterval &LI = LIS->getInterval(VirtReg);

    ArrayRef<MCPhysReg> PrefOrder = TRC.getRawAllocationOrder(*MF);

    PhysRegSet.push_back(std::vector<MCPhysReg>());

    for (ArrayRef<MCPhysReg>::iterator PhysRegIt = PrefOrder.begin(); PhysRegIt != PrefOrder.end(); ++PhysRegIt) {
      if (!MRI->isAllocatable(*PhysRegIt)) {
        continue;
      }

      // TODO Check if this line is necessary
      if (LRM->checkInterference(LI, *PhysRegIt) != llvm::LiveRegMatrix::IK_Free) {
       continue;
      }

      PhysRegSet[VirtId].push_back(*PhysRegIt);
    }
    
    for (MachineRegisterInfo::def_instr_iterator DefIt = MRI->def_instr_begin(VirtReg); 
        DefIt != MRI->def_instr_end(); ++DefIt) {
      genInstrRestrictions(*DefIt, VirtId, LI, false);
    }

    for (MachineRegisterInfo::use_instr_iterator UseIt = MRI->use_instr_begin(VirtReg);
        UseIt != MRI->use_instr_end(); ++UseIt) {
      genInstrRestrictions(*UseIt, VirtId, LI, false);
    }

    for (llvm::LiveInterval::iterator SegIt = LI.begin(); SegIt != LI.end(); ++SegIt) {
      for (SlotIndex InstrIdx = SegIt->start.getBaseIndex(); InstrIdx <= SegIt->end; InstrIdx = InstrIdx.getNextIndex()) {
        MachineInstr *Instr = SIS->getInstructionFromIndex(InstrIdx);

        if (Instr == nullptr) {
          continue;
        }

        genInstrRestrictions(*Instr, VirtId, LI, true);
      }
    }
  }
}

void RegAllocQueue::genStateVaraibles(void) {
  for (MachineBasicBlock &BasicBlock : *MF) {
    if (BasicBlock.empty()) {
      continue; 
    }

    MachineInstrBundleIterator<MachineInstr> FirstInstrIt = BasicBlock.getFirstNonDebugInstr();
    if (FirstInstrIt == BasicBlock.end()) {
      continue;
    }
    SlotIndex FirstInstrIdx = SIS->getInstructionIndex(*FirstInstrIt);
    MoveQueue FirstQ = Queues[SlotIndexMap[FirstInstrIdx]];

    // TODO: Do first instruction stuff

    MoveQueue PrevQ = FirstQ;
    for (MachineInstrBundleIterator<MachineInstr> InstrIt = std::next(FirstInstrIt);
        InstrIt != BasicBlock.end(); ++InstrIt) {
      SlotIndex InstrIdx = SIS->getInstructionIndex(*InstrIt);
      size_t InstrId = SlotIndexMap[InstrIdx];
      
      MoveQueue CurQ = Queues[InstrId];

      for (virt_id VirtId : LivenessMap[InstrId]) {

        

        for (MCPhysReg PhysReg : PhysRegSet[VirtId]) {
          GRBVar* Vars = CurQ.find(std::pair{VirtId, PhysReg});

          for (size_t SlotId = 0; SlotId < QUEUE_SLOTS; ++SlotId) {
            
          }
        }
      }

      PrevQ = CurQ;
    }
  }
}

void RegAllocQueue::genClashRestrictions(void) {
  for (size_t LiveSetId = 0; LiveSetId < LivenessMap.size(); ++LiveSetId) {
    std::set<virt_id> LiveSet = LivenessMap[LiveSetId];

    for (std::set<virt_id>::iterator IdIt1 = LiveSet.begin(); IdIt1 != LiveSet.end(); ++IdIt1) {
      for (std::set<virt_id>::iterator IdIt2 = std::next(IdIt1, 1); IdIt2 != LiveSet.end(); ++IdIt2) {
        MoveQueue *Q = &Queues[LiveSetId];

        for (MCPhysReg PhysReg1 : PhysRegSet[*IdIt1]) {
          for (MCPhysReg PhysReg2 : PhysRegSet[*IdIt2]) {
            if (!TRI->regsOverlap(PhysReg1, PhysReg2)) {
              continue;
            }

            GRBVar* Vars1 = Q->find(std::pair{*IdIt1, PhysReg1});
            GRBVar* Vars2 = Q->find(std::pair{*IdIt2, PhysReg2});

            for (size_t SlotId = 0; SlotId < QUEUE_SLOTS; ++SlotId) {
              Model->addConstr(Vars1[SlotId] + Vars2[SlotId], GRB_LESS_EQUAL, 1, "Clash Restriction");
            }

            // TODO Add something for aliasing maybe

          }
        }
      }
    }
  }
}

void RegAllocQueue::genBlockRestrictions(void) {
  for (MachineBasicBlock &BasicBlock : *MF) {
    if (BasicBlock.empty()) {
      continue;
    }

    MachineInstrBundleIterator<MachineInstr> LastInstrIt = BasicBlock.getFirstTerminator();
    if (LastInstrIt == BasicBlock.end()) {
      LastInstrIt = BasicBlock.getLastNonDebugInstr();
    }
    SlotIndex LastInstrIdx = SIS->getInstructionIndex(*LastInstrIt).getBaseIndex();

    for (MachineBasicBlock *NextBlock : BasicBlock.successors()) {
      if (NextBlock->empty()) {
        continue;
      }

      MachineInstrBundleIterator<MachineInstr> FirstInstrIt = NextBlock->getFirstNonDebugInstr();
      SlotIndex FirstInstrIdx = SIS->getInstructionIndex(*FirstInstrIt).getBaseIndex();

      for (virt_id LastId : LivenessMap[SlotIndexMap[LastInstrIdx]]) {
        for (virt_id FirstId : LivenessMap[SlotIndexMap[FirstInstrIdx]]) {
          if (LastId != FirstId) {
            continue;
          }

          for (MCPhysReg LastPhysReg : PhysRegSet[LastId]) {
            for (MCPhysReg FirstPhysReg : PhysRegSet[FirstId]) {
              if (LastPhysReg != FirstPhysReg) {
                continue;
              }

              for (size_t SlotId = 0; SlotId < QUEUE_SLOTS; ++SlotId) {
                Model->addConstr();
              }
            }
          }
        }
      }
    }
  }
}

void RegAllocQueue::genTermsRestrictions(void) {

}

void RegAllocQueue::genObjectiveFunction(void) {

}

void RegAllocQueue::readPBOSolution(void) {

}

void RegAllocQueue::addBlockLiveIns(void) {

}

void RegAllocQueue::assignRegisters(void) {

}

GRBEnv SolverEnv = GRBEnv();

bool RegAllocQueue::runOnMachineFunction(llvm::MachineFunction &MachineFunction) {
  MF = &MachineFunction;
  LIS = &getAnalysis<llvm::LiveIntervals>();
  SIS = &getAnalysis<llvm::SlotIndexes>();
  LRM = &getAnalysis<llvm::LiveRegMatrix>();

  TII = MF->getSubtarget().getInstrInfo();
  TRI = MF->getSubtarget().getRegisterInfo();
  MRI = &MF->getRegInfo();

  MBFI = &getAnalysis<llvm::MachineBlockFrequencyInfo>();

  SlotIndexMap = std::map<SlotIndex, size_t>();
  LivenessMap = std::vector<std::set<virt_id>>();

  StackCache = std::set<Register>();
  VarCount = 0;

  ObjectiveExpr = GRBQuadExpr();

  GRBModel Solver = GRBModel(SolverEnv);
  Model = &Solver;
  Model->getEnv().set(GRB_DoubleParam_TimeLimit, 1200);

  genStoreRestrictions();
  genClashRestrictions();
  genBlockRestrictions();
  genTermsRestrictions();

  genObjectiveFunction();
  Model->setObjective(ObjectiveExpr);

  readPBOSolution();
  addBlockLiveIns();
  assignRegisters();

  MRI->clearVirtRegs();

  return true;
}

FunctionPass *llvm::createPBOQueueRegisterAllocator() {
  return new RegAllocQueue();
}

FunctionPass *llvm::createPBOQueueRegisterAllocator(RegClassFilterFunc F) {
  return new RegAllocQueue(F);
}
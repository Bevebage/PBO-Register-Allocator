#include "LiveDebugVariables.h"
#include "llvm/ADT/ArrayRef.h"
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
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocCommon.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
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
#include <sstream>
#include <string>
#include <sys/types.h>
#include <system_error>
#include <utility>
#include <vector>

#include "gurobi_c++.h"

using namespace llvm;

#define DEBUG_TYPE "regalloc"
#define SOLVER_COMMAND "gurobi_cl ResultFile=solution.sol TimeLimit=60"

static RegisterRegAlloc PboUpdatedRegAlloc("pbo-updated", "PBO updated register allocator", createPBOUpdatedRegisterAllocator);

typedef unsigned long long idx_t;
typedef std::map<SlotIndex, std::set<idx_t>> slot_queue_t;

namespace {

class PBORegAllocUpdated : public MachineFunctionPass {

  MachineFunction *MachineFunction;
  VirtRegMap *VirtRegMap;
  LiveIntervals *LiveIntervals;
  LiveRegMatrix *LiveRegMatrix;
  MachineRegisterInfo *MachineRegisterInfo;
  SlotIndexes *SlotIndexes;
  MachineBlockFrequencyInfo *MachineBlockFrequencyInfo;
  const TargetRegisterInfo *TargetRegisterInfo;
  const TargetInstrInfo *TargetInstrInfo;

  std::vector<Register> VirtRegs;
  idx_t VirtRegCount;

  slot_queue_t SlotQueue;

  std::ofstream OpbFile;
  std::fstream ConstraintFile;
  std::string ObjectiveFunction;

  GRBModel *Model;

  idx_t LogVarCount;
  idx_t ConstraintCount;

  std::map<Register, idx_t> VirtIdxes;

  std::vector<std::vector<MCPhysReg>> PhysSets;
  std::vector<std::map<SlotIndex, std::map<MCPhysReg, std::string>>> PhysLogVarSets;
  std::vector<std::map<SlotIndex, std::string>> SpillLogVarSets;

  std::map<std::string, bool> LogVarAssignments;
  std::vector<std::map<SlotIndex, MCPhysReg>> RegAssignments;
  std::map<Register, int> StackMapping;

  // Instruction Insertion Stuff
  std::map<MCPhysReg, MCPhysReg> DestToSrc;
  std::map<MCPhysReg, idx_t> DestToReg;

  std::vector<std::pair<MCPhysReg, idx_t>> ToZero;
  std::map<MCPhysReg, bool> Starts;
  std::set<MCPhysReg> Ends;

public:
  PBORegAllocUpdated(const RegClassFilterFunc F = allocateAllRegClasses);

  StringRef getPassName() const override { return "Updated PBO Register Allocator"; };

  void getAnalysisUsage(AnalysisUsage &AUsage) const override;

  bool runOnMachineFunction(class MachineFunction &MF) override;

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

  std::string genLogVar(void);
  void instrConditions(const MachineInstr &, idx_t, bool);

  void insertEmptyInstr(void);

  void genStoreConditions(void);
  void genLiveRestrictions(void);
  void genConflictConditions(void);
  void genEndingRestrictions(void);

  void mixSwapPenalty(void);

  void genAssignments();
  void substOperand(MachineBasicBlock::instr_iterator&, SlotIndex);
  void assignPhysRegisters(void);

  int getStackSlot(Register Reg);
  int getStackSlot(const TargetRegisterClass &);
  void addMBBLiveIns(void);

  static char ID;

  // Instruction Insertion Stuff
  void addEdge(std::pair<MCPhysReg, MCPhysReg>, idx_t);
  void insertInstrs(MachineBasicBlock &, MachineBasicBlock::instr_iterator &);
  void clearQueue(void);
};

char PBORegAllocUpdated::ID = 0;

} // namespace

INITIALIZE_PASS_BEGIN(PBORegAllocUpdated, "regallocpboupdated", 
                      "Updated PBO Allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(VirtRegMap)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_DEPENDENCY(LiveRegMatrix)
INITIALIZE_PASS_END(PBORegAllocUpdated, "regallocpboupdated", 
                    "Updated PBO Allocator", false, false)

PBORegAllocUpdated::PBORegAllocUpdated(RegClassFilterFunc F) : MachineFunctionPass(ID) {}

void PBORegAllocUpdated::getAnalysisUsage(AnalysisUsage &AUsage) const {
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

std::string PBORegAllocUpdated::genLogVar(void) {
  return "x" + std::to_string(++LogVarCount);
}

void PBORegAllocUpdated::instrConditions(const MachineInstr &Instr, idx_t VirtIdx, bool Spillable) {
  SlotIndex SegIdx = SlotIndexes->getInstructionIndex(Instr);

  slot_queue_t::iterator RegGroup = SlotQueue.find(SegIdx);

  if (RegGroup == SlotQueue.end()) {
    RegGroup = SlotQueue.insert({SegIdx, std::set<idx_t>({VirtIdx})}).first;
  } else if (RegGroup->second.find(VirtIdx) != RegGroup->second.end()) {
    return;
  }

  RegGroup->second.insert(VirtIdx);

  PhysLogVarSets[VirtIdx][SegIdx] = std::map<MCPhysReg, std::string>();

  idx_t PhysRegCount = PhysSets[VirtIdx].size();

  for (size_t PhysIdx = 0; PhysIdx < PhysRegCount; ++PhysIdx) {
    std::string LogVar = genLogVar();
    PhysLogVarSets[VirtIdx][SegIdx][PhysSets[VirtIdx][PhysIdx]] = LogVar;
    ObjectiveFunction += " -" + std::to_string((PhysRegCount - PhysIdx) / (float) PhysRegCount) + " " + LogVar;
    ConstraintFile << " +1 " << LogVar;
  }

  if (Spillable) {
    std::string LogVar = genLogVar();
    SpillLogVarSets[VirtIdx][SegIdx] = LogVar;
    ConstraintFile << " +1 " << LogVar;
  }

  ConstraintFile << " = 1;\n";
  ConstraintCount++;

  // Implicit Conditions

  for (MachineOperand Operand : Instr.operands()) {
    if (Operand.isRegMask()) {
      for (MCPhysReg PhysReg : PhysSets[VirtIdx]) {
        if (Operand.clobbersPhysReg(PhysReg)) {
          ConstraintFile << " +1 " << PhysLogVarSets[VirtIdx][SegIdx][PhysReg] << " = 0;\n";
          ConstraintCount++;
        }
      }
      continue;
    }
  }
}

void PBORegAllocUpdated::insertEmptyInstr(void) {
  for (MachineBasicBlock &MachineBasicBlock : *MachineFunction) {
    if (!MachineBasicBlock.empty()) {
      continue;
    }

    BuildMI(MachineBasicBlock, MachineBasicBlock.begin(), DebugLoc(), TargetInstrInfo->get(TargetOpcode::KILL));
    SlotIndexes->insertMachineInstrInMaps(MachineBasicBlock.instr_front());
  }
}

void PBORegAllocUpdated::genStoreConditions(void) {
  idx_t VirtIdx = 0;
  for (Register VirtReg : VirtRegs) {
    VirtIdxes[VirtReg] = VirtIdx;

    const TargetRegisterClass &RegisterClass = *MachineRegisterInfo->getRegClass(VirtReg);
    ArrayRef<uint16_t> RawOrder = RegisterClass.getRawAllocationOrder(*MachineFunction);

    LiveInterval &LiveInterval = LiveIntervals->getInterval(VirtReg);

    PhysSets.push_back(std::vector<MCPhysReg>());

    for (ArrayRef<uint16_t>::iterator PhysRegIt = RawOrder.begin(); PhysRegIt != RawOrder.end(); ++PhysRegIt) {
      if (MachineRegisterInfo->isReserved(*PhysRegIt)) {
        continue;
      }
      if (LiveRegMatrix->checkInterference(LiveInterval, *PhysRegIt) != llvm::LiveRegMatrix::IK_Free) {
        continue;
      }
      if (MachineRegisterInfo->isAllocatable(*PhysRegIt)) {
        PhysSets[VirtIdx].push_back(*PhysRegIt);
      } 
    }

    PhysLogVarSets.push_back(std::map<SlotIndex, std::map<MCPhysReg, std::string>>());
    SpillLogVarSets.push_back(std::map<SlotIndex, std::string>());

    for (MachineRegisterInfo::def_instr_iterator DefIt = MachineRegisterInfo->def_instr_begin(VirtReg);
         DefIt != MachineRegisterInfo::def_instr_end(); ++DefIt) {
      instrConditions(*DefIt, VirtIdx, false);
    }

    for (MachineRegisterInfo::use_instr_nodbg_iterator UseIt = MachineRegisterInfo->use_instr_nodbg_begin(VirtReg);
         UseIt != MachineRegisterInfo::use_instr_nodbg_end(); ++UseIt) {

      instrConditions(*UseIt, VirtIdx, false);
    }

    for (llvm::LiveInterval::iterator IntIt = LiveInterval.begin(); IntIt != LiveInterval.end(); ++IntIt) {
      // LLVM_DEBUG(dbgs() << "START: "; IntIt->start.print(dbgs()); dbgs() << " "; dbgs() << "END: "; IntIt->end.print(dbgs()); dbgs() << "\n");
      for (SlotIndex SegIdx = IntIt->start.getBaseIndex(); SegIdx <= IntIt->end; SegIdx = SegIdx.getNextIndex()) {
        MachineInstr *Instr = SlotIndexes->getInstructionFromIndex(SegIdx);

        if (Instr == nullptr) {
          continue;
        }

        instrConditions(*Instr, VirtIdx, true);
      }
    }

    VirtIdx++;
  }
}

void PBORegAllocUpdated::genConflictConditions(void) {
  ConstraintFile << "* Confliction Restrictions\n";
  for (slot_queue_t::iterator SlotIt = SlotQueue.begin(); SlotIt != SlotQueue.end(); ++SlotIt) {
    for (std::set<idx_t>::iterator RegIt1 = SlotIt->second.begin(); RegIt1 != SlotIt->second.end(); ++RegIt1) {
      for (std::set<idx_t>::iterator RegIt2 = std::next(RegIt1, 1); RegIt2 != SlotIt->second.end(); ++RegIt2) {
        for (MCPhysReg PhysReg1 : PhysSets[*RegIt1]) {
          for (MCPhysReg PhysReg2 : PhysSets[*RegIt2]) {
            if (TargetRegisterInfo->regsOverlap(PhysReg1, PhysReg2)) {
              ConstraintFile << " +1 " << PhysLogVarSets[*RegIt1][SlotIt->first][PhysReg1] 
                             << " +1 " << PhysLogVarSets[*RegIt2][SlotIt->first][PhysReg2]
                             << " <= 1;\n";

              ConstraintCount++;
            }
          }
        }
      }
    }
  }
}

void PBORegAllocUpdated::genEndingRestrictions(void) {
  ConstraintFile << "* Ending Restrictions\n";
  for (MachineBasicBlock &BasicBlock : *MachineFunction) {
    if (BasicBlock.empty()) {
      continue;
    }

    SlotIndex LastIdx = SlotIndexes->getInstructionIndex(*BasicBlock.getLastNonDebugInstr()).getBaseIndex();
    std::set<idx_t> *ExitLiveVirts = &SlotQueue[LastIdx];

    for (MachineBasicBlock *Successor : BasicBlock.successors()) {
      if (Successor->empty()) {
        continue;
      }

      LLVM_DEBUG(dbgs() << "inside: " << Successor->getName() << "\n");

      SlotIndex FirstIdx = SlotIndexes->getInstructionIndex(*Successor->getFirstNonDebugInstr()).getBaseIndex();
      std::set<idx_t> *EntryLiveVirts = &SlotQueue[FirstIdx];

      LLVM_DEBUG(dbgs() << "outside\n");

      for (idx_t ExitIdx : *ExitLiveVirts) {
        for (idx_t EntryIdx : *EntryLiveVirts) {
          if (ExitIdx != EntryIdx) {
            continue;
          }


          for (MCPhysReg PhysReg1 : PhysSets[ExitIdx]) {
            for (MCPhysReg PhysReg2 : PhysSets[EntryIdx]) {
              if (PhysReg1 != PhysReg2) {
                continue;
              }

              ConstraintFile << " +1 " << PhysLogVarSets[ExitIdx][LastIdx][PhysReg1]
                             << " -1 " << PhysLogVarSets[EntryIdx][FirstIdx][PhysReg2]
                             << " = 0;\n";
              ConstraintCount++;
            }
          }
        }
      }
    }
  }
}

void PBORegAllocUpdated::mixSwapPenalty(void) {
  for (MachineBasicBlock &MachineBasicBlock : *MachineFunction) {
    if (MachineBasicBlock.empty()) {
      continue;
    }

    MachineInstr &StartInstr = *MachineBasicBlock.getFirstNonDebugInstr();
    SlotIndex StartIdx = SlotIndexes->getInstructionIndex(StartInstr);
    double Expected = 20 * MachineBlockFrequencyInfo->getBlockFreqRelativeToEntryBlock(&MachineBasicBlock);

    std::map<idx_t, std::map<MCPhysReg, std::string>> PrevMap = {};
    std::map<idx_t, std::string> PrevSpill = {};

    for (idx_t VirtIdx : SlotQueue[StartIdx]) {
      for (MCPhysReg PhysReg : PhysSets[VirtIdx]) {
        PrevMap[VirtIdx][PhysReg] = PhysLogVarSets[VirtIdx][StartIdx][PhysReg];
      }

      if (SpillLogVarSets[VirtIdx].find(StartIdx) != SpillLogVarSets[VirtIdx].end()) {
        PrevSpill[VirtIdx] = SpillLogVarSets[VirtIdx][StartIdx];
      } 
    }

    llvm::MachineBasicBlock::instr_iterator InstIt = MachineBasicBlock.instr_begin();
    for (InstIt++; InstIt != MachineBasicBlock.instr_end(); ++InstIt) {
      if (InstIt->isDebugInstr()) {
        continue;
      }

      SlotIndex Idx = SlotIndexes->getInstructionIndex(*InstIt);

      std::map<idx_t, std::map<MCPhysReg, std::string>> CurMap = {};
      std::map<idx_t, std::string> CurSpill = {};

      for (idx_t VirtIdx : SlotQueue[Idx]) {
        if (PrevMap.find(VirtIdx) == PrevMap.end()) {
          CurMap[VirtIdx] = PhysLogVarSets[VirtIdx][Idx];
          continue;
        }

        for (MCPhysReg PhysReg : PhysSets[VirtIdx]) {
          ObjectiveFunction += " -" + std::to_string(Expected) + " " + PrevMap[VirtIdx][PhysReg] + " " + PhysLogVarSets[VirtIdx][Idx][PhysReg];
        }

        bool CanSpill = SpillLogVarSets[VirtIdx].find(Idx) != SpillLogVarSets[VirtIdx].end();

        if (CanSpill) {
          CurSpill[VirtIdx] =  SpillLogVarSets[VirtIdx][Idx];

          if (PrevSpill.find(VirtIdx) != PrevSpill.end()) {
            ObjectiveFunction += " -" + std::to_string(Expected / 2) + " " + PrevSpill[VirtIdx] + " " + CurSpill[VirtIdx];
          }
        }

        CurMap[VirtIdx] = PhysLogVarSets[VirtIdx][Idx];
      }

      PrevMap = CurMap;
      PrevSpill = CurSpill;
    }
  }
}

void PBORegAllocUpdated::genAssignments() {
  std::ifstream File("solution.sol");

  LLVM_DEBUG(dbgs() << "LOG_VAR_COUNT: " << LogVarCount << "\n");

  if (File.is_open()) {
    std::string Line;

    LLVM_DEBUG(
      std::string file_name = "/Users/pierreyan/logs/"+MachineFunction->getName().str()+".sol";
      dbgs() << "Writing to " << file_name << "\n";

      std::ofstream dst;
      dst.open(file_name, std::ios::out | std::ios::trunc);

      if (dst.is_open()) {
        dst << File.rdbuf();
        dst.close();
        File.seekg(0);
      } else {
        std::cerr << "Failed to open file: " << strerror(errno) << "\n";
      }
    );

    getline(File, Line);
    for (idx_t Reads = 0; Reads < LogVarCount && getline(File, Line); ++Reads) {
      std::stringstream Stream(Line);

      std::string Var, Assignment;
      Stream >> Var >> Assignment; 

      LLVM_DEBUG(dbgs() << "Var: " << Var << " Assignment: " << Assignment << "\n");
      LogVarAssignments[Var] = std::stoi(Assignment) != 0;
    }

    File.close();
  } else {
    LLVM_DEBUG(dbgs() << "Couldn't open solutions\n");
  }

  idx_t VirtIdx = 0;
  for (Register VirtReg : VirtRegs) {
    LiveInterval &LiveInterval = LiveIntervals->getInterval(VirtReg);

    RegAssignments.push_back(std::map<SlotIndex, MCPhysReg>());

    for (llvm::LiveInterval::iterator IntIt = LiveInterval.begin(); IntIt != LiveInterval.end(); ++IntIt) {
      for (SlotIndex SegIdx = IntIt->start.getBaseIndex(); SegIdx <= IntIt->end; SegIdx = SegIdx.getNextIndex()) {
        MachineInstr *Instr = SlotIndexes->getInstructionFromIndex(SegIdx);

        if (Instr == nullptr) {
          continue;
        }

        // LLVM_DEBUG(dbgs() << "INSTR: "; Instr->print(dbgs()));

        RegAssignments[VirtIdx][SegIdx] = 0;
        // LLVM_DEBUG(dbgs() << printReg(VirtRegs[VirtIdx]) << " <- ");
        for (MCPhysReg PhysReg : PhysSets[VirtIdx]) {
          if (LogVarAssignments[PhysLogVarSets[VirtIdx][SegIdx][PhysReg]]) {
            RegAssignments[VirtIdx][SegIdx] = PhysReg;
            // LLVM_DEBUG(dbgs() << printReg(PhysReg) << "\n");
            break;
          }
          // LLVM_DEBUG(dbgs() << "0\n");
        }
      }
    }

    VirtIdx++;
  }

  // for (auto Entry : SlotQueue) {
  //   LLVM_DEBUG(dbgs() << "INSTR: "; SlotIndexes->getInstructionFromIndex(Entry.first)->print(dbgs()));
  //   for (idx_t VirtIdx : Entry.second) {
  //     LLVM_DEBUG(dbgs() << printReg(VirtRegs[VirtIdx]) << " -> " << printReg(RegAssignments[VirtIdx][Entry.first]) << "\n");
  //   }
  // }
}

void PBORegAllocUpdated::substOperand(MachineBasicBlock::instr_iterator &Inst, SlotIndex InstIdx) {
  for (MachineOperand &Operand : (*Inst).operands()) {
    if (!Operand.isReg()) {
      continue;
    }

    Register Reg = Operand.getReg();

    if (!Register::isVirtualRegister(Reg)) {
      continue;
    }

    idx_t VirtIdx = VirtIdxes[Reg];

    Operand.substPhysReg(RegAssignments[VirtIdx][InstIdx], *TargetRegisterInfo);
  }
}

void PBORegAllocUpdated::addEdge(std::pair<MCPhysReg, MCPhysReg> Move, idx_t VirtIdx) {
  LLVM_DEBUG(dbgs() << "Move: " << printReg(Move.first) << " -> " << printReg(Move.second) << "\n");
  if (Move.second == 0) {
    ToZero.push_back(std::pair<MCPhysReg, idx_t>{Move.first, VirtIdx});
    return;
  }

  Starts[Move.second] = false;

  if (Starts.find(Move.first) != Starts.end()) {
    Starts[Move.first] = true;

    if (Ends.find(Move.second) != Ends.end()) {
      MCPhysReg TermReg = Move.second;

      MCPhysReg CurReg = Move.first;
      while (DestToSrc.find(CurReg) != DestToSrc.end()) {
        CurReg = DestToSrc[CurReg];
      }

      if (CurReg != TermReg) {
        Starts[Move.second] = true;
      }
    }
  } else {
    if (Ends.find(Move.second) != Ends.end()) {
      Starts[Move.second] = true;
    }
  }

  Ends.insert(Move.first);

  DestToSrc[Move.second] = Move.first;
  DestToReg[Move.second] = VirtIdx;
}

void PBORegAllocUpdated::insertInstrs(MachineBasicBlock &MachineBasicBlock, llvm::MachineBasicBlock::instr_iterator &InsertIt) {
  LLVM_DEBUG(dbgs() << "START: "; InsertIt->print(dbgs()));
  // LLVM_DEBUG(dbgs() << "TO_ZERO\n");
  for (std::pair<MCPhysReg, idx_t> Blunt : ToZero) {
    Register VirtReg = VirtRegs[Blunt.second];
    int FrameIdx = getStackSlot(VirtReg);
    const TargetRegisterClass &RegisterClass = *MachineRegisterInfo->getRegClass(VirtReg);

    TargetInstrInfo->storeRegToStackSlot(MachineBasicBlock, InsertIt, Blunt.first, true, FrameIdx, &RegisterClass, TargetRegisterInfo, Register());
    SlotIndexes->insertMachineInstrInMaps(*std::prev(InsertIt));
    // LLVM_DEBUG(dbgs() << printReg(Blunt.first) << " -> " << "0" << "\n");

    MCPhysReg CurReg = Blunt.first;
    while (DestToSrc.find(CurReg) != DestToSrc.end()) {
      MCPhysReg SrcReg = DestToSrc[CurReg];
      
      if (SrcReg != 0) {
        TargetInstrInfo->copyPhysReg(MachineBasicBlock, InsertIt, InsertIt->getDebugLoc(), CurReg, SrcReg, true);
        SlotIndexes->insertMachineInstrInMaps(*std::prev(InsertIt));
        // LLVM_DEBUG(dbgs() << printReg(SrcReg) << " -> " << printReg(CurReg) << "\n");
      } else {
        VirtReg = VirtRegs[DestToReg[CurReg]];
        FrameIdx = getStackSlot(VirtReg);
        const TargetRegisterClass &RegisterClass = *MachineRegisterInfo->getRegClass(VirtReg);

        TargetInstrInfo->loadRegFromStackSlot(MachineBasicBlock, InsertIt, CurReg, FrameIdx, &RegisterClass, TargetRegisterInfo, Register());
        SlotIndexes->insertMachineInstrInMaps(*std::prev(InsertIt));
        // LLVM_DEBUG(dbgs() << "0" << " -> " << printReg(CurReg) << "\n");
      }

      Starts[CurReg] = true;
      CurReg = SrcReg;
    }
  }

  // LLVM_DEBUG(dbgs() << "STACK\n");
  for (std::pair<MCPhysReg, bool> Start : Starts) {
    if (Start.second) {
      continue; 
    }
    std::vector<MCPhysReg> Stack = {};

    MCPhysReg CurReg = Start.first; bool Cycle = false;
    while (DestToSrc.find(CurReg) != DestToSrc.end()) {
      Starts[CurReg] = true;
      Stack.push_back(CurReg);
      MCPhysReg SrcReg = DestToSrc[CurReg];

      if (SrcReg == 0) {
        break;
      }

      if (SrcReg == Start.first) {
        Cycle = true;
        break;
      }

      CurReg = SrcReg;
    }

    if (Cycle) {
      // LLVM_DEBUG(dbgs() << "CYCLE!\n");
      Register VirtReg = VirtRegs[DestToReg[Start.first]];
      int FrameIdx = getStackSlot(*MachineRegisterInfo->getRegClass(VirtReg));
      const TargetRegisterClass &RegisterClass = *MachineRegisterInfo->getRegClass(VirtReg);

      TargetInstrInfo->storeRegToStackSlot(MachineBasicBlock, InsertIt, Start.first, true, FrameIdx, &RegisterClass, TargetRegisterInfo, Register());
      SlotIndexes->insertMachineInstrInMaps(*std::prev(InsertIt));
      // LLVM_DEBUG(dbgs() << printReg(Start.first) << " -> " << "0" << "\n");

      for (idx_t Idx = 0; Idx < Stack.size() - 1; ++Idx) {
        TargetInstrInfo->copyPhysReg(MachineBasicBlock, InsertIt, InsertIt->getDebugLoc(), Stack[Idx], Stack[Idx+1], true);
        SlotIndexes->insertMachineInstrInMaps(*std::prev(InsertIt));
        // LLVM_DEBUG(dbgs() << printReg(Stack[Idx+1]) << " -> " << printReg(Stack[Idx]) << "\n");
      }

      MCPhysReg LastReg = Stack[Stack.size()-1];

      VirtReg = VirtRegs[DestToReg[LastReg]];
      const TargetRegisterClass &LastRegisterClass = *MachineRegisterInfo->getRegClass(VirtReg);

      TargetInstrInfo->loadRegFromStackSlot(MachineBasicBlock, InsertIt, LastReg, FrameIdx, &LastRegisterClass, TargetRegisterInfo, Register());
      SlotIndexes->insertMachineInstrInMaps(*std::prev(InsertIt));
      // LLVM_DEBUG(dbgs() << "0" << " -> " << printReg(LastReg) << "\n");
    } else {
      for (idx_t Idx = 0; Idx < Stack.size() - 1; ++Idx) {
        TargetInstrInfo->copyPhysReg(MachineBasicBlock, InsertIt, InsertIt->getDebugLoc(), Stack[Idx], Stack[Idx+1], true);
        SlotIndexes->insertMachineInstrInMaps(*std::prev(InsertIt));
        // LLVM_DEBUG(dbgs() << printReg(Stack[Idx+1]) << " -> " << printReg(Stack[Idx]) << "\n");
      }
      
      MCPhysReg LastReg = Stack[Stack.size()-1];
      // LLVM_DEBUG(dbgs() << "LAST REG: " << printReg(LastReg) << " SOURCE: " << printReg(DestToSrc[LastReg]) << "\n");
      if (DestToSrc[LastReg] == 0) {
        Register VirtReg = VirtRegs[DestToReg[LastReg]];
        int FrameIdx = getStackSlot(VirtReg);
        const TargetRegisterClass &RegisterClass = *MachineRegisterInfo->getRegClass(VirtReg);

        TargetInstrInfo->loadRegFromStackSlot(MachineBasicBlock, InsertIt, LastReg, FrameIdx, &RegisterClass, TargetRegisterInfo, Register());
        SlotIndexes->insertMachineInstrInMaps(*std::prev(InsertIt));
        // LLVM_DEBUG(dbgs() << "0" << " -> " << printReg(LastReg) << "\n");
      } else {
        TargetInstrInfo->copyPhysReg(MachineBasicBlock, InsertIt, InsertIt->getDebugLoc(), LastReg, DestToSrc[LastReg], true);
        SlotIndexes->insertMachineInstrInMaps(*std::prev(InsertIt));
        // LLVM_DEBUG(dbgs() << printReg(DestToSrc[LastReg]) << " -> " << printReg(LastReg) << "\n");

      }
    }
  }

}

void PBORegAllocUpdated::clearQueue(void) {
  DestToSrc = {};
  DestToReg = {};

  ToZero = {};
  Starts = {};
  Ends = {};
}

void PBORegAllocUpdated::assignPhysRegisters(void) {
  for (llvm::MachineBasicBlock &MachineBasicBlock : (*MachineFunction)) {
    std::map<idx_t, MCPhysReg> LiveMap = {};

    for (llvm::MachineBasicBlock::instr_iterator InstIt = MachineBasicBlock.instr_begin();
         InstIt != MachineBasicBlock.instr_end(); ++InstIt) {

      if (InstIt->isDebugInstr()) {
        continue;
      }

      SlotIndex InstIdx = SlotIndexes->getInstructionIndex(*InstIt).getBaseIndex();

      substOperand(InstIt, InstIdx);

      // This is to ensure that we do not break up the terminator block at the end of a basic block
      MachineBasicBlock::instr_iterator InsertIt = InstIt;
      while (InsertIt->isTerminator() && InsertIt != MachineBasicBlock.instr_begin() && std::prev(InsertIt)->isTerminator()) {
        InsertIt--;
      }

      clearQueue();
      std::map<idx_t, MCPhysReg> NewMap = {};

      // Moving values to and from stack slots or between registers, as needed
      for (idx_t VirtIdx : SlotQueue[InstIdx]) {
        NewMap[VirtIdx] = RegAssignments[VirtIdx][InstIdx];

        // If new
        if (LiveMap.find(VirtIdx) == LiveMap.end()) {
          continue;
        }

        // If nothing changes
        if (RegAssignments[VirtIdx][InstIdx] == LiveMap[VirtIdx]) {
          continue;
        }

        addEdge(std::pair<MCPhysReg, MCPhysReg>{LiveMap[VirtIdx], RegAssignments[VirtIdx][InstIdx]}, VirtIdx);
        // LLVM_DEBUG(dbgs() << "ADD: " << printReg(LiveMap[VirtIdx]) << " -> " << printReg(RegAssignments[VirtIdx][InstIdx]) << "\n");
      }

      LiveMap = NewMap;

      insertInstrs(MachineBasicBlock, InsertIt);
    }
  }

  // Delete live ranges
}

int PBORegAllocUpdated::getStackSlot(Register Reg) {
  if (StackMapping[Reg] != -1) {
    return StackMapping[Reg];
  }

  const TargetRegisterClass &RegisterClass = *MachineRegisterInfo->getRegClass(Reg);

  idx_t Size = TargetRegisterInfo->getSpillSize(RegisterClass);
  Align Alignment = TargetRegisterInfo->getSpillAlign(RegisterClass);

  int FrameIdx = MachineFunction->getFrameInfo().CreateSpillStackObject(Size, Alignment);

  StackMapping[Reg] = FrameIdx;
  return FrameIdx;
}

int PBORegAllocUpdated::getStackSlot(const TargetRegisterClass & RegisterClass) {
  idx_t Size = TargetRegisterInfo->getSpillSize(RegisterClass);
  Align Alignment = TargetRegisterInfo->getSpillAlign(RegisterClass);

  int FrameIdx = MachineFunction->getFrameInfo().CreateSpillStackObject(Size, Alignment);

  return FrameIdx;
}

void PBORegAllocUpdated::addMBBLiveIns(void) {
  for (MachineBasicBlock &MachineBasicBlock : (*MachineFunction)) {
    if (MachineBasicBlock.empty()) {
      continue;
    }

    SlotIndex StartIdx = SlotIndexes->getInstructionIndex(*MachineBasicBlock.getFirstNonDebugInstr());

    for (idx_t VirtIdx : SlotQueue[StartIdx]) {
      MCPhysReg PhysReg = RegAssignments[VirtIdx][StartIdx];

      if (PhysReg == 0) {
        continue;
      }

      MachineBasicBlock.addLiveIn(PhysReg);
    }
  }
}

bool PBORegAllocUpdated::runOnMachineFunction(class MachineFunction &MF) {
  MachineFunction = &MF;
  VirtRegMap = &getAnalysis<llvm::VirtRegMap>();
  LiveIntervals = &getAnalysis<llvm::LiveIntervals>();
  LiveRegMatrix = &getAnalysis<llvm::LiveRegMatrix>();
  TargetRegisterInfo = MachineFunction->getSubtarget().getRegisterInfo();
  TargetInstrInfo = MachineFunction->getSubtarget().getInstrInfo();
  MachineRegisterInfo = &MachineFunction->getRegInfo();
  SlotIndexes = &getAnalysis<llvm::SlotIndexes>();
  MachineBlockFrequencyInfo = &getAnalysis<llvm::MachineBlockFrequencyInfo>();

  VirtRegs = std::vector<Register>();
  VirtRegCount = MachineRegisterInfo->getNumVirtRegs();

  StackMapping = std::map<Register, int>();

  for (idx_t Idx = 0; Idx < VirtRegCount; ++Idx) {
    Register VirtReg = Register::index2VirtReg(Idx);

    VirtRegs.push_back(VirtReg);
    StackMapping[VirtReg] = -1;
  }

  LogVarCount = 0;
  ConstraintCount = 0;

  ObjectiveFunction = "min:";

  std::string OpbFileName = MachineFunction->getName().str() + "_problem.opb";
  std::string ConFileName = MachineFunction->getName().str() + "_constraints.txt";

  OpbFile.open(OpbFileName, std::ofstream::out | std::ofstream::trunc);
  ConstraintFile.open(ConFileName, std::ofstream::out | std::ofstream::trunc);

  SlotQueue = slot_queue_t();
  VirtIdxes = std::map<Register, idx_t>();

  PhysSets = std::vector<std::vector<MCPhysReg>>();
  PhysLogVarSets = std::vector<std::map<SlotIndex, std::map<MCPhysReg, std::string>>>();
  SpillLogVarSets = std::vector<std::map<SlotIndex, std::string>>();

  GRBModel Solver = GRBModel({});
  Model = &Solver;

  genStoreConditions();
  genConflictConditions();
  genEndingRestrictions();

  mixSwapPenalty();

  OpbFile << "*#variable= " << LogVarCount
          << " #constraint= " << ConstraintCount
          << "\n* constraints for function " << MachineFunction->getName().str()
          << "\n";
  OpbFile << ObjectiveFunction << ";\n";

  ConstraintFile.close();
  std::ifstream ConstraintInFile;
  ConstraintInFile.open(ConFileName);
  OpbFile << ConstraintInFile.rdbuf();
  ConstraintInFile.close();
  OpbFile.close();

  std::string SolverCmd(SOLVER_COMMAND);
  std::string Command;

  if (false) { // Timeout goes here
    // Command = SolverCmd = " Default " + std::to_string(Timeout) + " " + OpbFileName
  } else {
    Command = SolverCmd + " " + OpbFileName + " > /dev/null";
  }

  system(Command.c_str());

  LogVarAssignments = std::map<std::string, bool>();
  RegAssignments = std::vector<std::map<SlotIndex, MCPhysReg>>();

  genAssignments();
  addMBBLiveIns();
  assignPhysRegisters();

  for (Register VirtReg : VirtRegs) {
    LiveIntervals->removeInterval(VirtReg);
  }

  MachineRegisterInfo->clearVirtRegs();

  return true;
}

FunctionPass *llvm::createPBOUpdatedRegisterAllocator() {
  return new PBORegAllocUpdated();
}

FunctionPass *llvm::createPBOUpdatedRegisterAllocator(RegClassFilterFunc F) {
  return new PBORegAllocUpdated(F);
}
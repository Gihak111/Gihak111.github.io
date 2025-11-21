class CustomCalendar extends HTMLElement {
  connectedCallback() {
    this.attachShadow({ mode: 'open' });
    
    // Expose current date to window for task checking
    window.currentCalendarDate = this.currentDate;
this.shadowRoot.innerHTML = `
      <style>
        .calendar-container {
          width: 300px;
          margin: 1rem auto;
          padding: 1rem;
          border: 1px solid #e5e7eb;
          border-radius: 0.5rem;
          background: white;
        }
        .calendar-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 0.5rem;
        }
        .calendar-title {
          font-weight: bold;
          color: #4b5563;
        }
        .calendar-body {
          display: grid;
          grid-template-columns: repeat(7, 1fr);
          gap: 0.5rem;
        }
        .calendar-day {
          text-align: center;
          padding: 0.5rem;
          font-size: 0.875rem;
          color: #6b7280;
        }
        .calendar-date {
          text-align: center;
          padding: 0.5rem;
          border-radius: 50%;
          cursor: pointer;
        }
        .calendar-date:hover {
          background-color: #e5e7eb;
        }
        .calendar-date.today {
          background-color: #6366f1;
          color: white;
        }
        .calendar-date.has-task {
          background-color: #a5b4fc;
          color: white;
        }
        .calendar-weekday {
          font-weight: bold;
          color: #4b5563;
          text-align: center;
          padding: 0.25rem;
          font-size: 0.75rem;
        }
      </style>
      <div class="calendar-container">
        <div class="calendar-header">
          <button id="prev-month"><i data-feather="chevron-left"></i></button>
          <div class="calendar-title" id="month-year"></div>
          <button id="next-month"><i data-feather="chevron-right"></i></button>
        </div>
        <div class="calendar-body" id="calendar-dates">
          <div class="calendar-weekday">일</div>
          <div class="calendar-weekday">월</div>
          <div class="calendar-weekday">화</div>
          <div class="calendar-weekday">수</div>
          <div class="calendar-weekday">목</div>
          <div class="calendar-weekday">금</div>
          <div class="calendar-weekday">토</div>
        </div>
      </div>
    `;

    this.currentDate = new Date();
    this.renderCalendar();
    this.setupEventListeners();
  }

  renderCalendar() {
    const monthYear = this.shadowRoot.getElementById('month-year');
    const calendarDates = this.shadowRoot.getElementById('calendar-dates');

    // Set month and year title
    const months = ['1월', '2월', '3월', '4월', '5월', '6월', '7월', '8월', '9월', '10월', '11월', '12월'];
    monthYear.textContent = `${months[this.currentDate.getMonth()]} ${this.currentDate.getFullYear()}`;

    // Clear previous dates
    while (calendarDates.children.length > 7) {
      calendarDates.removeChild(calendarDates.lastChild);
    }

    // Get first day of month and total days in month
    const firstDay = new Date(this.currentDate.getFullYear(), this.currentDate.getMonth(), 1);
    const lastDay = new Date(this.currentDate.getFullYear(), this.currentDate.getMonth() + 1, 0);
    const totalDays = lastDay.getDate();
    const firstDayIndex = firstDay.getDay();

    // Add empty cells for days before first day
    for (let i = 0; i < firstDayIndex; i++) {
      const emptyCell = document.createElement('div');
      emptyCell.className = 'calendar-date';
      calendarDates.appendChild(emptyCell);
    }

    // Add cells for each day of the month
    const today = new Date();
    for (let i = 1; i <= totalDays; i++) {
      const dateCell = document.createElement('div');
      dateCell.className = 'calendar-date';
      dateCell.textContent = i;
      
      // Highlight today
      if (i === today.getDate() && this.currentDate.getMonth() === today.getMonth() && this.currentDate.getFullYear() === today.getFullYear()) {
        dateCell.classList.add('today');
      }
        // Check if date has tasks
        const hasTasks = window.tasks?.some(task => {
            if (!task.dueDate) return false;
            const taskDate = new Date(task.dueDate);
            return (
                taskDate.getDate() === i &&
                taskDate.getMonth() === this.currentDate.getMonth() &&
                taskDate.getFullYear() === this.currentDate.getFullYear()
            );
        });

        if (hasTasks) {
            dateCell.classList.add('has-task');
        }
calendarDates.appendChild(dateCell);
    }
  }

  setupEventListeners() {
    this.shadowRoot.getElementById('prev-month').addEventListener('click', () => {
      this.currentDate.setMonth(this.currentDate.getMonth() - 1);
      this.renderCalendar();
      feather.replace();
    });

    this.shadowRoot.getElementById('next-month').addEventListener('click', () => {
      this.currentDate.setMonth(this.currentDate.getMonth() + 1);
      this.renderCalendar();
      feather.replace();
    });
  }
}

customElements.define('custom-calendar', CustomCalendar);